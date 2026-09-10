interface EmptyStateProps {
  title: string;
  detail: string;
}

/** Honest empty state — used wherever real data genuinely doesn't exist yet. */
export function EmptyState({ title, detail }: EmptyStateProps) {
  return (
    <div className="flex flex-col items-center justify-center rounded-lg border border-dashed border-border bg-surface px-6 py-16 text-center">
      <p className="text-sm font-medium text-text-primary">{title}</p>
      <p className="mt-1 max-w-md text-xs text-text-secondary">{detail}</p>
    </div>
  );
}
