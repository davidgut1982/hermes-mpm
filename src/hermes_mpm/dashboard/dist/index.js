/**
 * MPM Runs dashboard plugin — frontend (hand-written plain IIFE).
 *
 * Why: The dashboard host serves this file verbatim (no build toolchain — no
 * npm/vite) and exposes React + UI primitives + an auth-aware fetch on
 * ``window.__HERMES_PLUGIN_SDK__``. Writing the panel as a plain IIFE that
 * pulls everything from that SDK keeps the plugin dependency-free and lets it
 * ride the host's React instance, auth, and design system. It mirrors the
 * structure of the bundled kanban plugin (same globals, same ``const API``
 * pattern, same ``register`` call).
 *
 * What: Renders an ``MpmRunsPage`` tab — status summary chips, a status filter
 * <select>, and a table of subagent runs (short id, status badge, profile/role,
 * age, duration, truncated goal) — polling ``/api/plugins/mpm-runs/runs`` every
 * 5s. Registers itself with the host plugin registry under "mpm-runs".
 *
 * Test: Not unit-tested here (the host React runtime isn't available in the
 * Python test harness). Verified by loading the dashboard tab after the symlink
 * + ``hermes-dashboard.service`` restart documented in DEPLOY_NOTES.md.
 */
(function () {
  "use strict";

  var SDK = window.__HERMES_PLUGIN_SDK__;
  var REGISTRY = window.__HERMES_PLUGINS__;
  if (!SDK || !REGISTRY) {
    // Host SDK not present — nothing to register against. Fail quiet.
    console.error("[mpm-runs] Hermes plugin SDK not found on window; aborting.");
    return;
  }

  var React = SDK.React;
  var h = React.createElement;
  var useState = SDK.hooks.useState;
  var useEffect = SDK.hooks.useEffect;
  var useCallback = SDK.hooks.useCallback;
  var fetchJSON = SDK.fetchJSON;
  var Badge = SDK.components.Badge;
  var timeAgo = (SDK.utils && SDK.utils.timeAgo) || function (ms) { return String(ms); };

  // Backend route base. Host ``fetchJSON`` prefixes origin + handles auth.
  var API = "/api/plugins/mpm-runs";
  var POLL_MS = 5000;
  var GOAL_MAX = 80;

  // Status vocabulary mirrors runs_db. Order here drives the chip row order.
  var STATUSES = ["running", "done", "failed", "crashed", "timed_out"];

  // Map a run status to a Badge variant + label. Badge variants are the host
  // design-system set (default/secondary/destructive/outline); we pick
  // sensible mappings and never assume a variant the host might not expose.
  var STATUS_META = {
    running: { variant: "default", label: "running" },
    done: { variant: "secondary", label: "done" },
    failed: { variant: "destructive", label: "failed" },
    crashed: { variant: "destructive", label: "crashed" },
    timed_out: { variant: "outline", label: "timed out" },
  };

  // ---- formatting helpers -------------------------------------------------

  // Short run id: runs are keyed by a session id; show the leading 8 chars so
  // the table stays scannable. Defensive against null/short ids.
  function shortId(id) {
    if (!id) return "—";
    return String(id).slice(0, 8);
  }

  // runs_db stores epoch SECONDS; the host timeAgo wants epoch MILLISECONDS.
  function ageOf(startedAt) {
    if (!startedAt) return "—";
    return timeAgo(Number(startedAt) * 1000);
  }

  // Human duration. Prefer the stored duration_ms; for a still-running row,
  // derive a live elapsed from the server ``now`` (epoch seconds) so the
  // column ticks without trusting the client clock.
  function durationOf(run, nowSec) {
    var ms = null;
    if (run.duration_ms != null) {
      ms = Number(run.duration_ms);
    } else if (run.status === "running" && run.started_at != null && nowSec != null) {
      ms = (Number(nowSec) - Number(run.started_at)) * 1000;
    }
    if (ms == null || isNaN(ms) || ms < 0) return "—";
    var s = Math.floor(ms / 1000);
    if (s < 60) return s + "s";
    var m = Math.floor(s / 60);
    var rem = s % 60;
    if (m < 60) return m + "m " + rem + "s";
    var hr = Math.floor(m / 60);
    return hr + "h " + (m % 60) + "m";
  }

  function profileRole(run) {
    var p = run.profile || "";
    var r = run.role || "";
    if (p && r && p !== r) return p + " / " + r;
    return p || r || "—";
  }

  function truncateGoal(goal) {
    if (!goal) return "—";
    var g = String(goal).replace(/\s+/g, " ").trim();
    if (g.length <= GOAL_MAX) return g;
    return g.slice(0, GOAL_MAX - 1) + "…";
  }

  function statusBadge(status) {
    var meta = STATUS_META[status] || { variant: "outline", label: status || "?" };
    return h(Badge, { variant: meta.variant }, meta.label);
  }

  // ---- summary chips ------------------------------------------------------

  function SummaryChips(props) {
    var stats = props.stats || {};
    var active = props.active;
    var onPick = props.onPick;

    var chips = STATUSES.map(function (st) {
      var count = stats[st] != null ? stats[st] : 0;
      var meta = STATUS_META[st] || { label: st };
      var isActive = active === st;
      return h(
        "button",
        {
          key: st,
          type: "button",
          onClick: function () { onPick(isActive ? "" : st); },
          className:
            "mpm-chip" + (isActive ? " mpm-chip-active" : ""),
          style: {
            display: "inline-flex",
            alignItems: "center",
            gap: "0.4rem",
            padding: "0.25rem 0.6rem",
            marginRight: "0.4rem",
            borderRadius: "0.5rem",
            border: "1px solid var(--border, #2a2a2a)",
            background: isActive
              ? "var(--accent, #2563eb)"
              : "var(--card, transparent)",
            color: isActive ? "#fff" : "inherit",
            cursor: "pointer",
            fontSize: "0.85rem",
          },
        },
        meta.label,
        h(
          "span",
          { style: { fontWeight: 600, opacity: 0.85 } },
          String(count)
        )
      );
    });

    return h(
      "div",
      { style: { display: "flex", flexWrap: "wrap", marginBottom: "0.75rem" } },
      chips
    );
  }

  // ---- runs table ---------------------------------------------------------

  function RunsTable(props) {
    var runs = props.runs || [];
    var nowSec = props.now;

    if (runs.length === 0) {
      return h(
        "div",
        {
          style: {
            padding: "2rem",
            textAlign: "center",
            opacity: 0.6,
          },
        },
        "No runs match the current filter."
      );
    }

    var headerCells = ["Run", "Status", "Profile / Role", "Age", "Duration", "Goal"].map(
      function (label) {
        return h(
          "th",
          {
            key: label,
            style: {
              textAlign: "left",
              padding: "0.4rem 0.6rem",
              borderBottom: "1px solid var(--border, #2a2a2a)",
              fontWeight: 600,
              fontSize: "0.8rem",
              opacity: 0.8,
            },
          },
          label
        );
      }
    );

    var bodyRows = runs.map(function (run) {
      var cellStyle = {
        padding: "0.4rem 0.6rem",
        borderBottom: "1px solid var(--border, #1d1d1d)",
        fontSize: "0.85rem",
        verticalAlign: "top",
      };
      return h(
        "tr",
        { key: run.run_id || Math.random() },
        h(
          "td",
          { style: cellStyle, title: run.run_id || "" },
          h("code", null, shortId(run.run_id))
        ),
        h("td", { style: cellStyle }, statusBadge(run.status)),
        h("td", { style: cellStyle }, profileRole(run)),
        h("td", { style: cellStyle }, ageOf(run.started_at)),
        h("td", { style: cellStyle }, durationOf(run, nowSec)),
        h("td", { style: cellStyle, title: run.goal || "" }, truncateGoal(run.goal))
      );
    });

    return h(
      "table",
      { style: { width: "100%", borderCollapse: "collapse" } },
      h("thead", null, h("tr", null, headerCells)),
      h("tbody", null, bodyRows)
    );
  }

  // ---- main page ----------------------------------------------------------

  function MpmRunsPage() {
    var runsState = useState([]);
    var runs = runsState[0];
    var setRuns = runsState[1];

    var statsState = useState({});
    var stats = statsState[0];
    var setStats = statsState[1];

    var nowState = useState(null);
    var now = nowState[0];
    var setNow = nowState[1];

    var filterState = useState("");
    var statusFilter = filterState[0];
    var setStatusFilter = filterState[1];

    var errState = useState(null);
    var error = errState[0];
    var setError = errState[1];

    var loadingState = useState(true);
    var loading = loadingState[0];
    var setLoading = loadingState[1];

    // Single refresh: pull both the filtered runs list and the stats aggregate.
    var refresh = useCallback(
      function () {
        var qs = statusFilter ? "?status=" + encodeURIComponent(statusFilter) : "";
        return Promise.all([
          fetchJSON(API + "/runs" + qs),
          fetchJSON(API + "/runs/stats"),
        ])
          .then(function (res) {
            var runsResp = res[0] || {};
            var statsResp = res[1] || {};
            setRuns(runsResp.runs || []);
            setNow(runsResp.now != null ? runsResp.now : null);
            setStats(statsResp.stats || {});
            setError(null);
            setLoading(false);
          })
          .catch(function (e) {
            setError(e && e.message ? e.message : String(e));
            setLoading(false);
          });
      },
      [statusFilter]
    );

    // Poll on mount + whenever the filter changes; clean up the interval.
    useEffect(
      function () {
        var cancelled = false;
        function tick() {
          if (cancelled) return;
          refresh();
        }
        tick();
        var id = setInterval(tick, POLL_MS);
        return function () {
          cancelled = true;
          clearInterval(id);
        };
      },
      [refresh]
    );

    var children = [];

    children.push(
      h(
        "div",
        {
          key: "head",
          style: {
            display: "flex",
            alignItems: "center",
            justifyContent: "space-between",
            marginBottom: "0.75rem",
          },
        },
        h("h2", { style: { margin: 0, fontSize: "1.1rem", fontWeight: 600 } }, "MPM Runs"),
        h(
          "label",
          { style: { fontSize: "0.85rem", display: "flex", gap: "0.4rem", alignItems: "center" } },
          "Status",
          h(
            "select",
            {
              value: statusFilter,
              onChange: function (e) { setStatusFilter(e.target.value); },
              style: {
                padding: "0.2rem 0.4rem",
                borderRadius: "0.4rem",
                background: "var(--card, transparent)",
                color: "inherit",
                border: "1px solid var(--border, #2a2a2a)",
              },
            },
            h("option", { value: "" }, "all"),
            STATUSES.map(function (st) {
              return h("option", { key: st, value: st }, st);
            })
          )
        )
      )
    );

    children.push(
      h(SummaryChips, {
        key: "chips",
        stats: stats,
        active: statusFilter,
        onPick: setStatusFilter,
      })
    );

    if (error) {
      children.push(
        h(
          "div",
          {
            key: "err",
            style: {
              padding: "0.6rem 0.8rem",
              marginBottom: "0.75rem",
              borderRadius: "0.4rem",
              border: "1px solid var(--destructive, #b91c1c)",
              color: "var(--destructive, #f87171)",
              fontSize: "0.85rem",
            },
          },
          "Failed to load runs: " + error
        )
      );
    }

    if (loading && runs.length === 0 && !error) {
      children.push(
        h(
          "div",
          { key: "loading", style: { padding: "2rem", textAlign: "center", opacity: 0.6 } },
          "Loading runs…"
        )
      );
    } else {
      children.push(h(RunsTable, { key: "table", runs: runs, now: now }));
    }

    return h(
      "div",
      { className: "mpm-runs-page", style: { padding: "1rem" } },
      children
    );
  }

  REGISTRY.register("mpm-runs", MpmRunsPage);
})();
