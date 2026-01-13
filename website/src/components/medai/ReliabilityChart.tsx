"use client";

import { Bar } from "react-chartjs-2";
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  BarElement,
  LineElement,
  PointElement,
  Tooltip,
  Legend,
  LineController,
} from "chart.js";

ChartJS.register(
  CategoryScale,
  LinearScale,
  BarElement,
  LineElement,
  PointElement,
  LineController,
  Tooltip,
  Legend
);

interface Props {
  prob_true: number[];
  prob_pred: number[];
  medicalLight?: boolean;
}

export default function ReliabilityChart({ prob_true, prob_pred, medicalLight }: Props) {
  const isLight = !!medicalLight;
  const labels = prob_pred.map((_, i) => `Bin ${i + 1}`);

  const data = {
    labels,
    datasets: [
      {
        label: "Predicted",
        data: prob_pred.map((v) => v * 100),
        backgroundColor: isLight ? "rgba(6,78,59,0.85)" : "rgba(14, 116, 144, 0.85)",
  LineElement,
  PointElement,
      },
      {
        type: "line",
        label: "Observed (true)",
        data: prob_true.map((v) => v * 100),
ChartJS.register(CategoryScale, LinearScale, BarElement, LineElement, PointElement, LineController, Tooltip, Legend);
    responsive: true,
    plugins: {
  bins: number[]; // predicted probability bin mids
  predicted: number[]; // predicted frequency
  observed: number[]; // observed frequency
  medicalLight?: boolean;
    scales: {
      x: {
export default function ReliabilityChart({ bins, predicted, observed, medicalLight }: Props) {
        ticks: { color: isLight ? '#374151' : '#9CA3AF', font: { size: 12 } },
  const labels = bins.map((b) => `${Math.round(b * 100)}%`);
      },
      y: {
        beginAtZero: true,
        max: 100,
        ticks: {
          callback: (v: any) => `${v}%`,
        data: predicted,
        backgroundColor: isLight ? 'rgba(14,116,144,0.9)' : 'rgba(99,102,241,0.9)',
        },
        grid: { color: isLight ? 'rgba(15,23,42,0.04)' : 'rgba(148,163,184,0.06)' },
      },
    },
        data: observed,
        borderColor: isLight ? 'rgba(6,78,59,0.9)' : 'rgba(14,116,144,0.9)',
}
