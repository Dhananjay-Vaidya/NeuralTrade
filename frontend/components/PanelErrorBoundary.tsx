"use client";

import { Component, type ReactNode, type ErrorInfo } from "react";
import { RefreshCw } from "lucide-react";

interface Props {
  children: ReactNode;
  /** Label shown in the fallback header, e.g. "Price Chart". */
  label: string;
  /**
   * Minimum height of the error fallback container — should match the content
   * height of the panel so the card occupies the same space as when loaded.
   * Pass a Tailwind class string like "min-h-[380px]".
   */
  minHeight: string;
}

interface State {
  error: Error | null;
}

export class PanelErrorBoundary extends Component<Props, State> {
  constructor(props: Props) {
    super(props);
    this.state = { error: null };
  }

  static getDerivedStateFromError(error: Error): State {
    return { error };
  }

  componentDidCatch(error: Error, info: ErrorInfo) {
    // Surface to console so the error isn't silently swallowed.
    console.error(`[PanelErrorBoundary] ${this.props.label}:`, error, info.componentStack);
  }

  private handleRetry = () => {
    this.setState({ error: null });
  };

  render() {
    const { error } = this.state;
    if (!error) return this.props.children;

    const { label, minHeight } = this.props;
    return (
      <div
        className={`flex flex-col items-center justify-center gap-3 ${minHeight} rounded-lg border border-red-500/20 bg-red-500/5 p-6 text-center`}
        role="alert"
        aria-label={`${label} failed to render`}
      >
        <p className="text-xs font-medium uppercase tracking-wide text-red-400/70">
          {label}
        </p>
        <p className="max-w-xs font-mono text-xs text-red-400/90">
          {error.message || "Unexpected render error"}
        </p>
        <button
          type="button"
          onClick={this.handleRetry}
          className="mt-1 flex items-center gap-1.5 rounded-lg border border-red-500/30 bg-red-500/10 px-3 py-1.5 text-xs text-red-300 transition-colors hover:bg-red-500/20 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-red-400"
        >
          <RefreshCw className="h-3 w-3" aria-hidden />
          Retry
        </button>
      </div>
    );
  }
}
