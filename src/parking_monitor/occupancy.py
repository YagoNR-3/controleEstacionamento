from typing import List
import time


class OccupancyStateMachine:
    def __init__(self, num_slots: int, occupy_confirm_s: float = 3.0, free_confirm_s: float = 5.5):
        self.num_slots = num_slots
        self.state: List[str] = ["free"] * num_slots
        self.changed_at: List[float] = [0.0] * num_slots
        self.occupy_confirm_s = occupy_confirm_s
        self.free_confirm_s = free_confirm_s

    def update_slot(self, index: int, detected: bool, now: float | None = None) -> None:
        if now is None:
            now = time.time()
        current = self.state[index]
        if current == "free":
            if detected:
                self.state[index] = "pending_occupied"
                self.changed_at[index] = now
        elif current == "pending_occupied":
            if detected:
                if (now - self.changed_at[index]) >= self.occupy_confirm_s:
                    self.state[index] = "occupied"
                    self.changed_at[index] = now
            else:
                self.state[index] = "free"
        elif current == "occupied":
            if not detected:
                self.state[index] = "pending_free"
                self.changed_at[index] = now
        elif current == "pending_free":
            if not detected:
                if (now - self.changed_at[index]) >= self.free_confirm_s:
                    self.state[index] = "free"
                    self.changed_at[index] = now
            else:
                self.state[index] = "occupied"

    def mark_detected_occupied_on_first_frame(self, detected_indices: set[int], now: float | None = None) -> None:
        if now is None:
            now = time.time()
        for i in range(self.num_slots):
            if i in detected_indices:
                self.state[i] = "occupied"
                self.changed_at[i] = now

    def count_occupied(self) -> int:
        return sum(s in ("occupied", "pending_free") for s in self.state)
