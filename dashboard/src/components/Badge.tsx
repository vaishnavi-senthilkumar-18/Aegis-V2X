interface BadgeProps {
  children: React.ReactNode;
  color?: string;
}

export function Badge({ children, color }: BadgeProps) {
  return (
    <span
      className="mono inline-flex items-center rounded-full border border-border bg-surface-raised px-2 py-0.5 text-[11px]"
      style={color ? { color, borderColor: color } : undefined}
    >
      {children}
    </span>
  );
}
