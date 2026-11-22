"""The UI listens to named SSE events; every name the code can emit must be on its list."""

from tests.event_names import emitted_event_names, frontend_event_names


def test_frontend_listens_to_every_emitted_event_name() -> None:
    emitted = emitted_event_names()
    assert len(emitted) > 50  # the scan found the call sites (ctx.event, append_event, ...)
    assert emitted - frontend_event_names() == set()


def test_scan_finds_split_and_indirect_call_sites() -> None:
    emitted = emitted_event_names()
    # ctx.event( split over lines; append_event's 4th argument; a constant; _end_before_running's argument
    assert {"node_error", "run_queued", "worker_claimed", "plan_rejected", "run_cancelled"} <= emitted
