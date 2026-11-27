//! Shared scheduling policy for terrain chunk streaming.
//!
//! The network representation of a terrain chunk is deliberately not part of
//! this module. These helpers only decide *when* and *in which order* already
//! available chunk work should be scheduled.

use super::TerrainChunkSize;
use crate::vol::RectVolSize;
use core::time::Duration;
use vek::*;

/// How far ahead of the player we look when ranking prefetch candidates.
pub const DEFAULT_PREFETCH_LOOKAHEAD: Duration = Duration::from_secs(2);
/// At most this fraction of the view radius is used as a look-ahead window.
pub const MAX_PREFETCH_VIEW_FRACTION: f32 = 0.75;
/// Number of missing chunks requested in one client tick.
pub const DEFAULT_REQUEST_BUDGET_PER_TICK: usize = 4;
/// Upper bound on outstanding client-side chunk requests.
pub const DEFAULT_MAX_PENDING_REQUESTS: usize = 24;
/// Maximum number of dead chunks retained only for shadow stability.
pub const DEFAULT_MAX_SHADOW_CHUNKS: usize = 32;
/// Soft time limit for GPU uploads performed while accepting mesh results.
pub const DEFAULT_MESH_UPLOAD_BUDGET: Duration = Duration::from_millis(2);
/// Hard cap on mesh results consumed in one frame.
pub const DEFAULT_MAX_MESH_UPLOADS_PER_FRAME: usize = 2;
/// Default peak estimate for CPU/GPU terrain mesh data.
pub const DEFAULT_MEMORY_BUDGET_BYTES: usize = 1024 * 1024 * 1024;

#[derive(Clone, Copy, Debug)]
pub struct TerrainStreamConfig {
    pub prefetch_lookahead: Duration,
    pub request_budget_per_tick: usize,
    pub max_pending_requests: usize,
    pub max_shadow_chunks: usize,
    pub mesh_upload_budget: Duration,
    pub max_mesh_uploads_per_frame: usize,
    pub memory_budget_bytes: usize,
}

impl Default for TerrainStreamConfig {
    fn default() -> Self {
        Self {
            prefetch_lookahead: DEFAULT_PREFETCH_LOOKAHEAD,
            request_budget_per_tick: DEFAULT_REQUEST_BUDGET_PER_TICK,
            max_pending_requests: DEFAULT_MAX_PENDING_REQUESTS,
            max_shadow_chunks: DEFAULT_MAX_SHADOW_CHUNKS,
            mesh_upload_budget: DEFAULT_MESH_UPLOAD_BUDGET,
            max_mesh_uploads_per_frame: DEFAULT_MAX_MESH_UPLOADS_PER_FRAME,
            memory_budget_bytes: DEFAULT_MEMORY_BUDGET_BYTES,
        }
    }
}

/// Offset the streaming focus in the direction of travel.
///
/// The offset is intentionally capped to a fraction of the view radius: a very
/// fast player should not make the scheduler request chunks that are not yet
/// useful (and would be rejected by the server anyway).
pub fn prefetch_offset(
    velocity_wpos: Vec2<f32>,
    view_distance: f32,
    lookahead: Duration,
) -> Vec2<f32> {
    let speed = velocity_wpos.magnitude();
    if speed <= f32::EPSILON {
        return Vec2::zero();
    }

    let chunk_size = TerrainChunkSize::RECT_SIZE.x as f32;
    let max_offset = (view_distance.max(0.0) * MAX_PREFETCH_VIEW_FRACTION * chunk_size).max(0.0);
    let desired_offset = speed * lookahead.as_secs_f32();
    velocity_wpos.normalized() * desired_offset.min(max_offset)
}

/// Position used to rank terrain work. At rest this is the player position; at
/// speed it moves ahead along the velocity vector.
pub fn prefetch_focus(
    focus_wpos: Vec2<f32>,
    velocity_wpos: Vec2<f32>,
    view_distance: f32,
    lookahead: Duration,
) -> Vec2<f32> {
    focus_wpos + prefetch_offset(velocity_wpos, view_distance, lookahead)
}

/// Lower is more urgent. Both the current and predicted distance are
/// normalized by the view radius, so the score remains meaningful when the
/// user changes view distance.
pub fn prefetch_priority(
    chunk_pos: Vec2<i32>,
    focus_wpos: Vec2<f32>,
    velocity_wpos: Vec2<f32>,
    view_distance: f32,
    lookahead: Duration,
) -> f32 {
    let chunk_size = TerrainChunkSize::RECT_SIZE.x as f32;
    let chunk_center = TerrainChunkSize::center_wpos(chunk_pos).as_::<f32>();
    let view_radius = (view_distance.max(1.0) * chunk_size).max(1.0);
    let predicted_focus = prefetch_focus(focus_wpos, velocity_wpos, view_distance, lookahead);

    let current_score = chunk_center.distance(focus_wpos) / view_radius;
    let predicted_score = chunk_center.distance(predicted_focus) / view_radius;
    let speed_ratio =
        (velocity_wpos.magnitude() * lookahead.as_secs_f32() / view_radius).clamp(0.0, 1.0);

    // At rest both distance terms coincide, so the score remains monotonic in
    // distance / view distance. At speed the predicted term dominates, while
    // the current term keeps nearby chunks responsive.
    predicted_score + current_score * (0.25 - 0.2 * speed_ratio)
}

/// Prevent a burst of expensive uploads from consuming a whole frame.
///
/// The first task is always allowed so a permanently over-budget (but still
/// bounded) upload cannot starve streaming completely.
#[derive(Clone, Copy, Debug)]
pub struct FrameBudget {
    limit: Duration,
    max_tasks: usize,
    spent: Duration,
    tasks: usize,
}

impl FrameBudget {
    pub fn new(limit: Duration, max_tasks: usize) -> Self {
        Self {
            limit,
            max_tasks,
            spent: Duration::ZERO,
            tasks: 0,
        }
    }

    /// Whether another task may start in this frame.
    pub fn can_start_task(&self) -> bool {
        self.tasks == 0 || (self.tasks < self.max_tasks && self.spent < self.limit)
    }

    /// Record the measured cost after a task has finished.
    pub fn charge(&mut self, cost: Duration) {
        self.spent = self.spent.saturating_add(cost);
        self.tasks += 1;
    }

    pub fn spent(&self) -> Duration { self.spent }

    pub fn tasks(&self) -> usize { self.tasks }

    pub fn exceeded(&self) -> bool { self.spent > self.limit }
}

/// A budget that tracks current and peak usage. Charging beyond the limit is
/// reported to the caller so optional work (for example keeping dead chunks
/// for shadows) can be skipped without panicking.
#[derive(Clone, Copy, Debug)]
pub struct MemoryBudget {
    limit_bytes: usize,
    used_bytes: usize,
    peak_bytes: usize,
}

impl MemoryBudget {
    pub fn new(limit_bytes: usize) -> Self {
        Self {
            limit_bytes,
            used_bytes: 0,
            peak_bytes: 0,
        }
    }

    /// Try to reserve `bytes`. On failure usage is left unchanged.
    pub fn try_charge(&mut self, bytes: usize) -> bool {
        let Some(new_used) = self.used_bytes.checked_add(bytes) else {
            return false;
        };
        if new_used > self.limit_bytes {
            return false;
        }
        self.used_bytes = new_used;
        self.peak_bytes = self.peak_bytes.max(new_used);
        true
    }

    /// Reserve bytes even if that would exceed the limit.
    ///
    /// Required work (visible terrain) uses this path, but the overrun remains
    /// observable through [`Self::over_budget`].
    pub fn force_charge(&mut self, bytes: usize) {
        self.used_bytes = self.used_bytes.saturating_add(bytes);
        self.peak_bytes = self.peak_bytes.max(self.used_bytes);
    }

    pub fn release(&mut self, bytes: usize) {
        self.used_bytes = self.used_bytes.saturating_sub(bytes);
    }

    pub fn used_bytes(&self) -> usize { self.used_bytes }

    pub fn peak_bytes(&self) -> usize { self.peak_bytes }

    pub fn limit_bytes(&self) -> usize { self.limit_bytes }

    pub fn over_budget(&self) -> bool { self.used_bytes > self.limit_bytes }
}

#[cfg(test)]
mod tests {
    use super::*;

    const CHUNK_SIZE: f32 = TerrainChunkSize::RECT_SIZE.x as f32;

    fn focus(chunk: Vec2<i32>) -> Vec2<f32> { TerrainChunkSize::center_wpos(chunk).as_() }

    #[test]
    fn stationary_requests_prefer_near_chunks() {
        let focus = focus(Vec2::new(0, 0));
        let near = prefetch_priority(
            Vec2::new(1, 0),
            focus,
            Vec2::zero(),
            10.0,
            DEFAULT_PREFETCH_LOOKAHEAD,
        );
        let far = prefetch_priority(
            Vec2::new(8, 0),
            focus,
            Vec2::zero(),
            10.0,
            DEFAULT_PREFETCH_LOOKAHEAD,
        );
        assert!(near < far, "near={near}, far={far}");
    }

    #[test]
    fn flight_speed_requests_prefer_ahead() {
        let focus = focus(Vec2::new(0, 0));
        let velocity = Vec2::new(80.0, 0.0);
        let ahead = prefetch_priority(
            Vec2::new(6, 0),
            focus,
            velocity,
            10.0,
            DEFAULT_PREFETCH_LOOKAHEAD,
        );
        let behind = prefetch_priority(
            Vec2::new(-6, 0),
            focus,
            velocity,
            10.0,
            DEFAULT_PREFETCH_LOOKAHEAD,
        );
        assert!(ahead < behind, "ahead={ahead}, behind={behind}");
    }

    #[test]
    fn prefetch_offset_is_bounded_by_view_distance() {
        let velocity = Vec2::new(1000.0, 0.0);
        let offset = prefetch_offset(velocity, 10.0, DEFAULT_PREFETCH_LOOKAHEAD);
        assert_eq!(offset.x, 10.0 * MAX_PREFETCH_VIEW_FRACTION * CHUNK_SIZE);
        assert_eq!(offset.y, 0.0);
    }

    #[test]
    fn frame_budget_always_progresses_then_stops() {
        let mut budget = FrameBudget::new(Duration::from_millis(2), 2);
        assert!(budget.can_start_task());
        budget.charge(Duration::from_millis(1));
        assert!(budget.can_start_task());
        budget.charge(Duration::from_millis(2));
        assert!(!budget.can_start_task());
        assert_eq!(budget.tasks(), 2);
    }

    #[test]
    fn frame_budget_stops_after_one_over_budget_task() {
        let mut budget = FrameBudget::new(Duration::from_millis(2), 2);
        assert!(budget.can_start_task());
        budget.charge(Duration::from_millis(5));
        assert!(!budget.can_start_task());
        assert!(budget.exceeded());
    }

    #[test]
    fn memory_budget_tracks_peak_and_rejects_overrun() {
        let mut budget = MemoryBudget::new(10);
        assert!(budget.try_charge(6));
        assert!(!budget.try_charge(5));
        assert_eq!(budget.used_bytes(), 6);
        budget.force_charge(6);
        assert_eq!(budget.used_bytes(), 12);
        assert_eq!(budget.peak_bytes(), 12);
        assert!(budget.over_budget());
        budget.release(8);
        assert_eq!(budget.used_bytes(), 4);
    }

    #[test]
    fn five_minute_flight_stays_within_upload_budget() {
        let config = TerrainStreamConfig::default();
        let costs = [
            Duration::from_micros(100),
            Duration::from_micros(300),
            Duration::from_micros(600),
            Duration::from_micros(900),
        ];

        for frame in 0..30 * 60 * 5 {
            let mut budget =
                FrameBudget::new(config.mesh_upload_budget, config.max_mesh_uploads_per_frame);
            let mut next_cost = frame;
            while budget.can_start_task() {
                budget.charge(costs[next_cost % costs.len()]);
                next_cost += 1;
            }
            assert!(
                !budget.exceeded(),
                "frame {frame} spent {:?}",
                budget.spent()
            );
            assert!(budget.tasks() <= config.max_mesh_uploads_per_frame);
        }
    }

    #[test]
    fn five_minute_flight_keeps_request_and_resident_counts_bounded() {
        // A lightweight deterministic model of the client request loop. It is not
        // a rendering benchmark; it guards the queue/memory invariants that can
        // be checked in CI.
        let config = TerrainStreamConfig::default();
        let view_distance = 10i32;
        let frames = 30 * 60 * 5;
        let chunk_size = CHUNK_SIZE;
        let velocity = Vec2::new(45.0, 0.0);
        let mut position = Vec2::new(0.0, 0.0);
        let mut pending = hashbrown::HashSet::new();
        let mut arrivals = Vec::new();
        let mut resident = hashbrown::HashSet::new();
        let mut peak_resident = 0usize;
        let mut peak_pending = 0usize;

        for frame in 0..frames {
            position += velocity / 30.0;
            let focus_chunk = (position / chunk_size).as_::<i32>();

            resident.retain(|key: &Vec2<i32>| {
                (key - focus_chunk)
                    .map(|e| e.unsigned_abs())
                    .magnitude_squared()
                    <= (view_distance as u32 + 3).pow(2)
            });

            for arrival in arrivals.drain(..).collect::<Vec<_>>() {
                pending.remove(&arrival);
                resident.insert(arrival);
            }

            let mut missing = Vec::new();
            for y in -view_distance..=view_distance {
                for x in -view_distance..=view_distance {
                    let key = focus_chunk + Vec2::new(x, y);
                    if (key - focus_chunk).map(|e| e as f32).magnitude() <= view_distance as f32
                        && !resident.contains(&key)
                        && !pending.contains(&key)
                    {
                        missing.push((
                            prefetch_priority(
                                key,
                                position,
                                velocity,
                                view_distance as f32,
                                config.prefetch_lookahead,
                            ),
                            key,
                        ));
                    }
                }
            }
            missing.sort_by(|a, b| a.0.total_cmp(&b.0));

            let slots = config
                .max_pending_requests
                .saturating_sub(pending.len())
                .min(config.request_budget_per_tick);
            let requests = missing.into_iter().take(slots).map(|(_, key)| key);
            for key in requests {
                pending.insert(key);
                // Deterministic but deliberately longer than one frame.
                arrivals.push(key);
                if frame % 3 == 0 && arrivals.len() > 30 {
                    let key = arrivals.remove(0);
                    pending.remove(&key);
                    resident.insert(key);
                }
            }

            peak_resident = peak_resident.max(resident.len());
            peak_pending = peak_pending.max(pending.len());
        }

        assert!(
            peak_pending <= config.max_pending_requests,
            "peak pending: {peak_pending}"
        );
        let resident_limit = (4 * (view_distance + 3).pow(2)) as usize;
        assert!(
            peak_resident <= resident_limit,
            "peak resident: {peak_resident}"
        );

        // A deliberately pessimistic fixed-size estimate. The real tracker uses
        // mesh/atlas sizes, but this validates the peak-memory invariant.
        const ESTIMATED_BYTES_PER_CHUNK: usize = 64 * 1024;
        assert!(
            peak_resident.saturating_mul(ESTIMATED_BYTES_PER_CHUNK) <= config.memory_budget_bytes,
            "estimated peak memory exceeds budget"
        );
    }
}
