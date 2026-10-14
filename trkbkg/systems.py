"""Detector system id -> human-readable name.

Source: the system table of eic2ai (src/system_ids.hpp, from the ePIC
definitions.xml). Same mapping that fills the trk_hit_system_name CSV column,
so we don't need to carry that string column through the parquet files.
Only ids that actually appear in tracker-hit data are listed; look up with
system_name() which falls back to the raw id.
"""

SYSTEM_NAMES = {
    25: "VertexSubAssembly_0", 26: "VertexSubAssembly_1", 27: "VertexSubAssembly_2",
    31: "VertexBarrel_0", 32: "VertexBarrel_1", 33: "VertexBarrel_2",
    34: "VertexEndcapN_0", 35: "VertexEndcapN_1", 36: "VertexEndcapN_2",
    37: "VertexEndcapP_0", 38: "VertexEndcapP_1", 39: "VertexEndcapP_2",
    59: "TrackerBarrel_0", 60: "TrackerBarrel_1", 61: "TrackerBarrel_2",
    62: "TrackerBarrel_3", 63: "TrackerBarrel_4", 64: "TrackerBarrel_5",
    65: "TrackerBarrel_6", 66: "TrackerBarrel_7", 67: "TrackerBarrel_8",
    68: "TrackerEndcapN_0", 69: "TrackerEndcapN_1", 70: "TrackerEndcapN_2",
    71: "TrackerEndcapN_3", 72: "TrackerEndcapN_4", 73: "TrackerEndcapN_5",
    74: "TrackerEndcapN_6", 75: "TrackerEndcapN_7", 76: "TrackerEndcapN_8",
    77: "TrackerEndcapP_0", 78: "TrackerEndcapP_1", 79: "TrackerEndcapP_2",
    80: "TrackerEndcapP_3", 81: "TrackerEndcapP_4", 82: "TrackerEndcapP_5",
    83: "TrackerEndcapP_6",
    90: "BarrelDIRC", 91: "BarrelTRD", 92: "BarrelTOF",
    120: "ForwardRICH", 121: "ForwardTRD", 122: "ForwardTOF",
    131: "BackwardRICH", 132: "BackwardTOF",
    150: "B0Tracker_Station_1", 151: "B0Tracker_Station_2",
    152: "B0Tracker_Station_3", 153: "B0Tracker_Station_4",
    154: "B0Preshower_Station_1",
    155: "ForwardRomanPot_Station_1", 156: "ForwardRomanPot_Station_2",
    159: "ForwardOffMTracker_station_1", 160: "ForwardOffMTracker_station_2",
    161: "ForwardOffMTracker_station_3", 162: "ForwardOffMTracker_station_4",
    193: "LumiSpecTracker", 198: "TaggerTracker",
}


def system_name(system_id) -> str:
    """'59 TrackerBarrel_0' — id kept in front so plots stay sortable."""
    sid = int(system_id)
    return f"{sid} {SYSTEM_NAMES.get(sid, '?')}"
