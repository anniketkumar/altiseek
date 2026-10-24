"""Sessions: groups of frames that must stay in one split (CLAUDE.md §7 step 5).

A session is one capture day across every camera group. The drones of a campaign filmed the same
exercise in the same hours (docs/data_notes.md), so splitting a day would put the same swimmers,
boats and sea state on both sides of a split. A day counts only if the frame's timestamp is
trusted: an image-level ``date_time``, or a MicaSense ``meta.date_time`` on a frame with a GPS fix.

Frames without a trusted day form one session per camera group, ``undated:<group>``: frames with
no timestamp at all (``trinity_rgb``) and MicaSense frames without GPS, whose camera clock dates
them to a day on which nothing else flew. Each session belongs to a campaign (site and year); an
undated session's campaign comes from the config.
"""

from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date

from altiseek.data.config import Site
from altiseek.data.frames import Frame


@dataclass(frozen=True)
class Session:
    """A group of frames that must share a split."""

    session_id: str
    campaign: str
    #: Capture day; None for undated sessions.
    day: date | None
    n_frames: int


def trusted_day(frame: Frame) -> date | None:
    """The frame's capture day, if its timestamp can be trusted."""
    if frame.timestamp is None:
        return None
    if frame.timestamp_source == "image":
        return frame.timestamp.date()
    if frame.timestamp_source == "meta" and frame.latitude_deg is not None:
        return frame.timestamp.date()
    return None


def session_of(frame: Frame) -> str:
    """Session id: the ISO capture day, or ``undated:<camera group>``."""
    day = trusted_day(frame)
    return day.isoformat() if day is not None else f"undated:{frame.camera_group}"


def build_sessions(
    frames: Sequence[Frame], sites: Sequence[Site], undated_campaign: Mapping[str, str]
) -> tuple[dict[int, str], dict[str, Session]]:
    """Session id per image id, and the sessions with their campaigns.

    A dated session's site is the one whose box contains its frames' GPS positions; every
    positioned frame of a day must fall in the same site, and every day needs at least one.
    """
    of = {f.image_id: session_of(f) for f in frames}
    members: defaultdict[str, list[Frame]] = defaultdict(list)
    for frame in frames:
        members[of[frame.image_id]].append(frame)
    sessions = {}
    for sid, group in sorted(members.items()):
        day = trusted_day(group[0])
        if day is None:
            camera_group = sid.removeprefix("undated:")
            if camera_group not in undated_campaign:
                raise ValueError(f"no campaign configured for undated session {sid}")
            campaign = undated_campaign[camera_group]
        else:
            campaign = f"{_site(sid, group, sites)}_{day.year}"
        sessions[sid] = Session(sid, campaign, day, len(group))
    return of, sessions


def _site(session_id: str, frames: Sequence[Frame], sites: Sequence[Site]) -> str:
    names = set()
    for frame in frames:
        if frame.latitude_deg is None or frame.longitude_deg is None:
            continue
        matches = [s.name for s in sites if s.contains(frame.latitude_deg, frame.longitude_deg)]
        if len(matches) != 1:
            raise ValueError(
                f"image {frame.image_id} at ({frame.latitude_deg}, {frame.longitude_deg}) "
                f"matches sites {matches}"
            )
        names.update(matches)
    if len(names) != 1:
        raise ValueError(f"session {session_id} has frames at sites {sorted(names)}")
    return names.pop()
