import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useState } from "react";
import { PageHeader } from "../components/PageHeader";
import { EmptyState } from "../components/EmptyState";
import { Badge } from "../components/Badge";
import { createSyntheticScene, listScenes, listVehicles } from "../lib/api";
import { formatVehicleLabel } from "../lib/format";

export function Simulation() {
  const queryClient = useQueryClient();
  const scenesQuery = useQuery({ queryKey: ["scenes"], queryFn: listScenes, refetchInterval: 5000 });
  const [selectedSceneId, setSelectedSceneId] = useState<string | null>(null);

  const [sceneCode, setSceneCode] = useState("IntersectionDemo01");
  const [numVehicles, setNumVehicles] = useState(5);
  const [numFrames, setNumFrames] = useState(200);

  const generateMutation = useMutation({
    mutationFn: createSyntheticScene,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["scenes"] });
    },
  });

  const scenes = scenesQuery.data ?? [];
  const activeSceneId = selectedSceneId ?? scenes[0]?.id ?? null;

  const vehiclesQuery = useQuery({
    queryKey: ["vehicles", activeSceneId],
    queryFn: () => listVehicles(activeSceneId as string),
    enabled: !!activeSceneId,
  });

  return (
    <div>
      <PageHeader
        title="Simulation"
        subtitle="Scene configuration and vehicle roster — synthetic generator stands in for Phase 2 (CARLA + Sionna RT)"
      />

      <div className="rounded-lg border border-border bg-surface p-5">
        <h2 className="text-sm font-medium text-text-primary">Generate Synthetic Scene</h2>
        <p className="mt-1 text-xs text-text-secondary">
          Produces a full scene: vehicles converging on a 4-way intersection, frames with
          physically plausible motion and channel state, and a complete trust/criticality/decision
          chain — through the same code path real ingestion would use.
        </p>
        <form
          className="mt-4 flex flex-wrap items-end gap-3"
          onSubmit={(e) => {
            e.preventDefault();
            generateMutation.mutate({ scene_code: sceneCode, num_vehicles: numVehicles, num_frames: numFrames });
          }}
        >
          <label className="flex flex-col gap-1 text-xs text-text-secondary">
            Scene Code
            <input
              className="rounded-md border border-border bg-bg px-2 py-1.5 text-sm text-text-primary"
              value={sceneCode}
              onChange={(e) => setSceneCode(e.target.value)}
              required
            />
          </label>
          <label className="flex flex-col gap-1 text-xs text-text-secondary">
            Vehicles
            <input
              type="number"
              min={1}
              max={16}
              className="w-24 rounded-md border border-border bg-bg px-2 py-1.5 text-sm text-text-primary"
              value={numVehicles}
              onChange={(e) => setNumVehicles(Number(e.target.value))}
            />
          </label>
          <label className="flex flex-col gap-1 text-xs text-text-secondary">
            Frames / Vehicle
            <input
              type="number"
              min={1}
              max={5000}
              className="w-28 rounded-md border border-border bg-bg px-2 py-1.5 text-sm text-text-primary"
              value={numFrames}
              onChange={(e) => setNumFrames(Number(e.target.value))}
            />
          </label>
          <button
            type="submit"
            disabled={generateMutation.isPending}
            className="rounded-md bg-accent px-4 py-1.5 text-sm font-medium text-white disabled:opacity-50"
          >
            {generateMutation.isPending ? "Generating…" : "Generate Scene"}
          </button>
        </form>
        {generateMutation.isSuccess && (
          <p className="mt-3 text-xs text-success">
            Created {generateMutation.data.scene_code}: {generateMutation.data.vehicles_created}{" "}
            vehicles, {generateMutation.data.frames_created} frames.
          </p>
        )}
        {generateMutation.isError && (
          <p className="mt-3 text-xs text-danger">{(generateMutation.error as Error).message}</p>
        )}
      </div>

      <div className="mt-6 grid grid-cols-1 gap-6 lg:grid-cols-[280px_1fr]">
        <div className="rounded-lg border border-border bg-surface p-4">
          <h2 className="text-sm font-medium text-text-primary">Scenes</h2>
          {scenes.length === 0 ? (
            <p className="mt-3 text-xs text-text-secondary">None yet.</p>
          ) : (
            <ul className="mt-3 space-y-1">
              {scenes.map((scene) => (
                <li key={scene.id}>
                  <button
                    type="button"
                    onClick={() => setSelectedSceneId(scene.id)}
                    className={`w-full rounded-md px-2 py-1.5 text-left text-sm ${
                      scene.id === activeSceneId
                        ? "bg-accent-soft text-text-primary"
                        : "text-text-secondary hover:bg-surface-raised"
                    }`}
                  >
                    {scene.scene_code}
                  </button>
                </li>
              ))}
            </ul>
          )}
        </div>

        <div className="rounded-lg border border-border bg-surface p-5">
          <h2 className="text-sm font-medium text-text-primary">Vehicle Roster</h2>
          {!activeSceneId ? (
            <div className="mt-4">
              <EmptyState title="Select a scene" detail="Pick a scene to see its vehicle roster." />
            </div>
          ) : (vehiclesQuery.data?.length ?? 0) === 0 ? (
            <div className="mt-4">
              <EmptyState title="No vehicles" detail="This scene has no vehicles registered." />
            </div>
          ) : (
            <table className="mt-4 w-full text-left text-xs">
              <thead>
                <tr className="text-text-secondary">
                  <th className="pb-2 pr-4">Vehicle</th>
                  <th className="pb-2 pr-4">Type</th>
                  <th className="pb-2 pr-4">Ego</th>
                </tr>
              </thead>
              <tbody className="mono">
                {vehiclesQuery.data?.map((vehicle) => (
                  <tr key={vehicle.id} className="border-t border-border">
                    <td className="py-1.5 pr-4">{formatVehicleLabel(vehicle.vehicle_code)}</td>
                    <td className="py-1.5 pr-4">{vehicle.vehicle_type}</td>
                    <td className="py-1.5 pr-4">
                      {vehicle.is_ego ? <Badge color="var(--color-accent)">ego</Badge> : "—"}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </div>
      </div>
    </div>
  );
}
