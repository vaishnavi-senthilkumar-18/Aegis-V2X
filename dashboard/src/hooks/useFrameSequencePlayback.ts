/**
 * Groups a scene's raw `Frame[]` (flat, one row per vehicle per
 * frame_index -- see `listFrames`) into an ordered sequence of "ticks",
 * each holding every vehicle's state at one frame_index, and drives
 * playback through that sequence on a timer.
 *
 * Real data note: frame_index values are NOT necessarily contiguous
 * integers (CARLA's tick counter, not an array index -- e.g. this
 * scene's real frame_index values are 121506, 121516, 121526, ...).
 * Never assume `frame_index` can be used directly as an array offset;
 * always go through the sorted `ticks` array this hook produces.
 *
 * Design-integrity note (matches api.ts's own stated rule): this hook
 * does not fabricate a position for any vehicle at a tick where it has
 * no real frame -- a vehicle simply does not appear in that tick's
 * `vehicles` map rather than being interpolated or invented.
 */

import { useEffect, useMemo, useRef, useState } from "react";
import { listFrames } from "../lib/api";
import type { Frame } from "../types/api";

export interface FrameTick {
  frameIndex: number;
  /** Keyed by vehicle_id (not vehicle_code) to match Frame's own shape. */
  vehicles: Record<string, Frame>;
}

export interface FrameSequence {
  ticks: FrameTick[];
  /** Convenience: every distinct vehicle_id seen anywhere in the sequence. */
  vehicleIds: string[];
}

function groupIntoTicks(frames: Frame[]): FrameSequence {
  const byFrameIndex = new Map<number, FrameTick>();
  const vehicleIdSet = new Set<string>();

  for (const frame of frames) {
    vehicleIdSet.add(frame.vehicle_id);
    let tick = byFrameIndex.get(frame.frame_index);
    if (!tick) {
      tick = { frameIndex: frame.frame_index, vehicles: {} };
      byFrameIndex.set(frame.frame_index, tick);
    }
    tick.vehicles[frame.vehicle_id] = frame;
  }

  const ticks = Array.from(byFrameIndex.values()).sort(
    (a, b) => a.frameIndex - b.frameIndex,
  );

  return { ticks, vehicleIds: Array.from(vehicleIdSet) };
}

export type PlaybackSpeed = 0.5 | 1 | 2;

export interface UseFrameSequencePlaybackResult {
  /** True while the initial fetch is in flight. */
  isLoading: boolean;
  /** Set if the fetch failed -- never silently swallowed (see api.ts's
   * own design-integrity comment: real errors surface, not fallback
   * data). */
  error: Error | null;
  /** The full grouped sequence, once loaded. Empty ticks array while loading. */
  sequence: FrameSequence;
  /** Index into `sequence.ticks` -- NOT a frame_index. */
  currentTickIndex: number;
  /** The tick currently being displayed, or null if there's nothing loaded yet. */
  currentTick: FrameTick | null;
  isPlaying: boolean;
  speed: PlaybackSpeed;
  play: () => void;
  pause: () => void;
  setSpeed: (speed: PlaybackSpeed) => void;
  /** Scrub directly to a tick index (e.g. from a timeline slider). */
  seekToTickIndex: (index: number) => void;
  /** Jump back to the first tick and stop. */
  reset: () => void;
}

//: Real scenes were exported at 10 Hz (see
//: `configs/simulation.yaml`'s sensor frequencies) -- one real-world
//: second of simulation per 10 ticks. At 1x speed, playback should
//: therefore advance one tick every 100ms to match real time, not an
//: arbitrarily chosen interval.
const MS_PER_TICK_AT_1X = 100;

export function useFrameSequencePlayback(
  sceneId: string | undefined,
): UseFrameSequencePlaybackResult {
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<Error | null>(null);
  const [sequence, setSequence] = useState<FrameSequence>({ ticks: [], vehicleIds: [] });
  const [currentTickIndex, setCurrentTickIndex] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);
  const [speed, setSpeed] = useState<PlaybackSpeed>(1);

  const intervalRef = useRef<ReturnType<typeof setInterval> | null>(null);

  // Fetch + group once per sceneId. Deliberately re-fetches (not cached
  // across scenes) since a scene's frame count can be large (thousands)
  // and holding multiple scenes' full sequences in memory at once isn't
  // needed for a single-scene playback view.
  useEffect(() => {
    if (!sceneId) {
      setSequence({ ticks: [], vehicleIds: [] });
      setIsLoading(false);
      return;
    }

    let cancelled = false;
    setIsLoading(true);
    setError(null);

    // limit=5000 covers this scene's real 4,960 frames with headroom;
    // revisit if/when a scene exceeds this (see FrameBulkCreate's own
    // 2000-per-request cap for the analogous ingest-side concern).
    listFrames({ scene_id: sceneId, limit: 5000 })
      .then((frames) => {
        if (cancelled) return;
        setSequence(groupIntoTicks(frames));
        setCurrentTickIndex(0);
      })
      .catch((err: Error) => {
        if (cancelled) return;
        setError(err);
      })
      .finally(() => {
        if (!cancelled) setIsLoading(false);
      });

    return () => {
      cancelled = true;
    };
  }, [sceneId]);

  // Playback timer.
  useEffect(() => {
    if (!isPlaying || sequence.ticks.length === 0) return;

    const intervalMs = MS_PER_TICK_AT_1X / speed;
    intervalRef.current = setInterval(() => {
      setCurrentTickIndex((prev) => {
        const next = prev + 1;
        if (next >= sequence.ticks.length) {
          // Reached the end: stop rather than loop silently, so the
          // person watching knows playback finished rather than
          // wondering if it's stuck.
          setIsPlaying(false);
          return prev;
        }
        return next;
      });
    }, intervalMs);

    return () => {
      if (intervalRef.current) clearInterval(intervalRef.current);
    };
  }, [isPlaying, speed, sequence.ticks.length]);

  const currentTick = useMemo(
    () => sequence.ticks[currentTickIndex] ?? null,
    [sequence.ticks, currentTickIndex],
  );

  return {
    isLoading,
    error,
    sequence,
    currentTickIndex,
    currentTick,
    isPlaying,
    speed,
    play: () => {
      // Restart from the beginning if playback had already reached the end.
      if (currentTickIndex >= sequence.ticks.length - 1) {
        setCurrentTickIndex(0);
      }
      setIsPlaying(true);
    },
    pause: () => setIsPlaying(false),
    setSpeed,
    seekToTickIndex: (index: number) => {
      const clamped = Math.max(0, Math.min(index, sequence.ticks.length - 1));
      setCurrentTickIndex(clamped);
    },
    reset: () => {
      setIsPlaying(false);
      setCurrentTickIndex(0);
    },
  };
}
