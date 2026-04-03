// src/pages/DataCleaning.js
import React, { useEffect, useState } from "react";
import Navbar from "../components/Navbar";
import {
  Chart as ChartJS,
  PointElement,
  CategoryScale,
  LinearScale,
  Tooltip,
  Legend,
} from "chart.js";
import { Scatter } from "react-chartjs-2";
import { useRef } from "react";
import jsPDF from "jspdf";
import html2canvas from "html2canvas";

ChartJS.register(PointElement, CategoryScale, LinearScale, Tooltip, Legend);

// 小型直方圖 SVG（真正的 histogram：用 bin center + bin width）
function HistogramSVG({ bins = [], counts = [], height = 160 }) {
  if (!bins || bins.length < 2 || !counts?.length) {
    return <div className="text-white/40 text-xs">無資料</div>;
  }

  const width = 220;
  const xMin = Math.min(...bins);
  const xMax = Math.max(...bins);
  const yMax = Math.max(...counts, 1);

  const mapX = (x) => ((x - xMin) / (xMax - xMin)) * width;
  const mapY = (c) => height - (c / yMax) * (height - 8);

  return (
    <svg
      width="100%"
      height={height}
      viewBox={`0 0 ${width} ${height}`}
      preserveAspectRatio="none"
    >
      {counts.map((c, i) => {
        const b0 = bins[i];
        const b1 = bins[i + 1];
        if (b1 === undefined) return null;

        const x0 = mapX(b0);
        const w0 = mapX(b1) - x0;

        const scale = 0.85;
        const w = w0 * scale;
        const x = x0 + (w0 - w) / 2;

        return (
          <rect
            key={i}
            x={x}
            y={mapY(c)}
            width={w}
            height={Math.max(0, height - mapY(c) - 4)}
            fill="#60a5fa"
            opacity="0.85"
          />
        );
      })}

      <line
        x1="0"
        x2={width}
        y1={height - 4}
        y2={height - 4}
        stroke="#555"
        strokeWidth="1"
      />
    </svg>
  );
}

// 箱型圖 SVG（支援排序 + 彩虹漸層顏色，匹配圖片，並支援「空箱細線」）
function BoxplotSVG({ groups = {}, width = 900, height = 400, expectedKeys }) {
  let keys;
  if (expectedKeys && expectedKeys.length > 0) {
    keys = expectedKeys.map(String);
  } else {
    keys = Object.keys(groups)
      .map(Number)
      .sort((a, b) => a - b)
      .map(String);
  }

  if (keys.length === 0) {
    return <div className="text-white/40 text-lg">無分組資料</div>;
  }

  let allVals = [];
  keys.forEach((k) => {
    const g = groups[k];
    if (!g) return;
    allVals.push(
      g.whisker_min ?? g.min,
      g.q1,
      g.median,
      g.q3,
      g.whisker_max ?? g.max,
      ...(g.outliers || [])
    );
  });

  if (!allVals.length || allVals.some((v) => v === undefined || v === null)) {
    return <div className="text-white/40 text-lg">無分組資料</div>;
  }

  const vmin = Math.min(...allVals);
  const vmax = Math.max(...allVals);
  const pad = (vmax - vmin) * 0.06 || 1;
  const rangeMin = vmin - pad;
  const rangeMax = vmax + pad;

  const mapY = (v) => {
    const hv = height - 60;
    return 30 + hv - ((v - rangeMin) / (rangeMax - rangeMin)) * hv;
  };

  const boxSlotWidth = (width - 80) / keys.length;
  const boxW = Math.max(12, boxSlotWidth * 0.6);
  const emptyLineY = height - 40;

  return (
    <svg
      width="100%"
      height={height}
      viewBox={`0 0 ${width} ${height}`}
      preserveAspectRatio="xMidYMid meet"
    >
      <rect x="0" y="0" width={width} height={height} fill="none" />

      {keys.map((k, i) => {
        const g = groups[k];
        const cx = 40 + i * boxSlotWidth + boxSlotWidth / 2;
        const boxLeft = cx - boxW / 2;
        const boxRight = cx + boxW / 2;

        const hue = keys.length > 1 ? (i / (keys.length - 1)) * 300 : 200;
        const boxColor = `hsl(${hue}, 80%, 60%)`;
        const lineColor = `hsl(${hue}, 70%, 40%)`;

        if (!g) {
          return (
            <g key={k}>
              <line
                x1={boxLeft}
                x2={boxRight}
                y1={emptyLineY}
                y2={emptyLineY}
                stroke="#555"
                strokeWidth={1}
                strokeDasharray="4 3"
              />
              <text
                x={cx}
                y={height - 10}
                textAnchor="middle"
                fontSize="12"
                fill="#ddd"
              >
                {k}
              </text>
            </g>
          );
        }

        const q1y = mapY(g.q1);
        const q3y = mapY(g.q3);
        const medy = mapY(g.median);
        const whiskerMiny = mapY(g.whisker_min ?? g.min);
        const whiskerMaxy = mapY(g.whisker_max ?? g.max);

        return (
          <g key={k}>
            <line
              x1={cx}
              x2={cx}
              y1={whiskerMaxy}
              y2={q3y}
              stroke={lineColor}
              strokeWidth={1.5}
            />
            <line
              x1={cx}
              x2={cx}
              y1={q1y}
              y2={whiskerMiny}
              stroke={lineColor}
              strokeWidth={1.5}
            />

            <line
              x1={boxLeft}
              x2={boxRight}
              y1={whiskerMaxy}
              y2={whiskerMaxy}
              stroke={lineColor}
              strokeWidth={1.5}
            />
            <line
              x1={boxLeft}
              x2={boxRight}
              y1={whiskerMiny}
              y2={whiskerMiny}
              stroke={lineColor}
              strokeWidth={1.5}
            />

            <rect
              x={boxLeft}
              y={q3y}
              width={boxW}
              height={Math.max(2, q1y - q3y)}
              fill={boxColor}
              opacity="0.4"
              stroke={lineColor}
              rx="2"
            />

            <line
              x1={boxLeft}
              x2={boxRight}
              y1={medy}
              y2={medy}
              stroke={lineColor}
              strokeWidth={3}
            />

            {g.outliers?.map((outlier, oi) => (
              <circle
                key={oi}
                cx={cx}
                cy={mapY(outlier)}
                r="3"
                fill="red"
                stroke="#900"
                strokeWidth="1"
              />
            ))}

            <text
              x={cx}
              y={height - 10}
              textAnchor="middle"
              fontSize="12"
              fill="#ddd"
            >
              {k}
            </text>
          </g>
        );
      })}
    </svg>
  );
}

// 相關性熱圖 SVG（美化版）
function CorrelationHeatmapSVG({ variables = [], matrix = [], width = 900, height = 960 }) {
  if (!variables.length || !matrix.length) {
    return <div className="text-white/40 text-lg">無相關性資料</div>;
  }

  const cellSize = (width - 120) / variables.length;
  const colorScale = (val) => {
    const abs = Math.abs(val);
    if (val >= 0) {
      const intensity = Math.min(abs, 1);
      return `rgb(${Math.floor(100 + 155 * intensity)}, ${Math.floor(
        150 + 105 * intensity
      )}, 255)`;
    } else {
      const intensity = Math.min(abs, 1);
      return `rgb(255, ${Math.floor(150 + 105 * (1 - intensity))}, ${Math.floor(
        150 + 105 * (1 - intensity)
      )})`;
    }
  };

  return (
    <svg
      width="100%"
      height={height}
      viewBox={`0 0 ${width} ${height}`}
      preserveAspectRatio="xMidYMid meet"
    >
      <rect x="0" y="0" width={width} height={height} fill="#1e1e1e" />

      {variables.map((rowVar, i) =>
        variables.map((colVar, j) => {
          const val = matrix[i][j];
          const x = 80 + j * cellSize;
          const y = 60 + i * cellSize;
          return (
            <g key={`${i}-${j}`}>
              <rect
                x={x}
                y={y}
                width={cellSize - 2}
                height={cellSize - 2}
                fill={colorScale(val)}
                stroke="#333"
                strokeWidth="1"
                rx="4"
              />
              <text
                x={x + cellSize / 2}
                y={y + cellSize / 2 + 4}
                textAnchor="middle"
                fontSize="11"
                fill="#000"
              >
                {val.toFixed(3)}
              </text>
            </g>
          );
        })
      )}

      {variables.map((varName, i) => (
        <text
          key={`row-${i}`}
          x={70}
          y={60 + i * cellSize + cellSize / 2 + 4}
          textAnchor="end"
          fontSize="12"
          fill="#ddd"
        >
          {varName}
        </text>
      ))}

      {variables.map((varName, j) => (
        <text
          key={`col-${j}`}
          x={80 + j * cellSize + cellSize / 2}
          y={40}
          textAnchor="middle"
          fontSize="12"
          fill="#ddd"
          transform={`rotate(-45 ${80 + j * cellSize + cellSize / 2} 40)`}
        >
          {varName}
        </text>
      ))}

      <text x={width / 2} y={20} textAnchor="middle" fontSize="16" fill="#fff">
        各特徵相關性熱圖 (Correlogram of PV Data)
      </text>
    </svg>
  );
}

// 散點圖（使用 Chart.js）
const AXIS_CONFIG = {
  EAC: { min: 0, max: 80, step: 20 },
  GI: { min: 0, max: 1000, step: 250 },
  TM: { min: 0, max: 60, step: 10 },
};

function RenderPairScatter({ rowVar, colVar, plots }) {
  const pairKey = `${colVar}__${rowVar}`;
  const pairData = plots?.pairs?.[pairKey];

  if (!pairData || !pairData.x || !pairData.y) {
    return <div className="text-white/40 text-xs">無資料</div>;
  }

  const points = pairData.x.map((x, idx) => ({
    x,
    y: pairData.y[idx],
    is_outlier: pairData.is_outlier ? pairData.is_outlier[idx] : false,
    index: idx + 1,
  }));

  const hasOutliers = points.some((p) => p.is_outlier);

  const chartData = {
    datasets: [
      {
        label: "正常值",
        data: points
          .filter((p) => !p.is_outlier)
          .map((p) => ({ x: p.x, y: p.y, idx: p.index })),
        backgroundColor: "rgba(96, 165, 250, 0.7)",
        pointRadius: 3,
      },
      ...(hasOutliers
        ? [
            {
              label: "離群值",
              data: points
                .filter((p) => p.is_outlier)
                .map((p) => ({ x: p.x, y: p.y, idx: p.index })),
              backgroundColor: "rgba(239, 68, 68, 0.9)",
              pointRadius: 5,
              pointStyle: "circle",
            },
          ]
        : []),
    ],
  };

  const xCfg = AXIS_CONFIG[colVar];
  const yCfg = AXIS_CONFIG[rowVar];

  const options = {
    maintainAspectRatio: false,
    plugins: {
      legend: { display: hasOutliers, position: "top", labels: { color: "#ddd" } },
      tooltip: {
        callbacks: {
          label: (context) => {
            const point = context.raw;
            const isOutlier = context.dataset.label === "離群值";
            const idx = point.idx;
            return `${context.dataset.label} (第 ${idx} 筆): (${point.x.toFixed(
              2
            )}, ${point.y.toFixed(2)})${isOutlier ? " ← 離群值" : ""}`;
          },
        },
      },
    },
    scales: {
      x: {
        type: "linear",
        ticks: { stepSize: xCfg?.step, color: "#aaa" },
        title: { display: true, text: colVar, color: "#ddd" },
      },
      y: {
        type: "linear",
        ticks: { stepSize: yCfg?.step, color: "#aaa" },
        title: { display: true, text: rowVar, color: "#ddd" },
      },
    },
  };

  return (
    <div className="relative h-64">
      <Scatter data={chartData} options={options} />
      <div className="absolute bottom-2 right-2 text-xs text-white/60">
        共 {points.length} 點{hasOutliers ? "（含離群值）" : ""}
      </div>
    </div>
  );
}

export default function DataCleaning({
  fileName: propFileName,
  onBack,
  onNext,
  onLogout,
  onNavigateToTrain,
  onNavigateToDashboard,
  onNavigateToSites,
  onOpenCreateSite,
  onNavigateToPredict,
  onNavigateToModelMgmt,
}) {
  const scatterRef = useRef();
  const boxplotRef = useRef();
  const corrRef = useRef();

  const pdfScatterRef = useRef();
  const pdfBoxplotRef = useRef();
  const pdfCorrRef = useRef();

  const [applyGiTm, setApplyGiTm] = useState(false);
  const [applyOutlier, setApplyOutlier] = useState(false);

  const [outlierMethod, setOutlierMethod] = useState("iqr_comprehensive");
  const [iqrFactor, setIqrFactor] = useState(2.0);
  const [zThreshold, setZThreshold] = useState(3.5);
  const [isolationContamination, setIsolationContamination] = useState(0.05);

  const [fileName] = useState(
    propFileName || localStorage.getItem("lastUploadedFile") || ""
  );

  const [siteId] = useState(
    localStorage.getItem("selectedSiteId") ||
      localStorage.getItem("lastSiteId") ||
      ""
  );

  const [stages, setStages] = useState(null);

  const rawHourKeys = stages?.raw?.boxplot_by_hour
    ? Object.keys(stages.raw.boxplot_by_hour)
        .map(Number)
        .sort((a, b) => a - b)
    : null;

  const currentStageKey = applyOutlier
    ? "after_outlier"
    : applyGiTm
    ? "after_gi_tm"
    : "raw";

  const plots = stages?.[currentStageKey] || null;

  const [loading, setLoading] = useState(false);
  const [saving, setSaving] = useState(false);

  const [result, setResult] = useState(null);
  const [showResultModal, setShowResultModal] = useState(false);

  const [selectedTab, setSelectedTab] = useState("scatter");
  const [selectedBoxplot, setSelectedBoxplot] = useState("month");

  useEffect(() => {
    if (!fileName || !siteId) return;

    const fetchData = async () => {
      setLoading(true);
      try {
        const params = new URLSearchParams({
          site_id: String(siteId),
          file_name: fileName,
          outlier_method: outlierMethod,
          iqr_factor: iqrFactor.toString(),
          z_threshold: zThreshold.toString(),
          isolation_contamination: isolationContamination.toString(),
          remove_outliers: applyOutlier.toString(),
          apply_gi_tm: applyGiTm.toString(),
        });

        const res = await fetch(
          `http://127.0.0.1:8000/visualize-data/?${params.toString()}`
        );

        if (!res.ok) {
          const text = await res.text();
          console.error("visualize-data error:", text);
          throw new Error("載入視覺化資料失敗");
        }

        const data = await res.json();
        setStages(data.stages);
      } catch (err) {
        console.error(err);
        alert("載入資料失敗，請確認 site_id 與檔案是否存在");
      } finally {
        setLoading(false);
      }
    };

    fetchData();
  }, [
    fileName,
    siteId,
    outlierMethod,
    iqrFactor,
    zThreshold,
    isolationContamination,
    applyGiTm,
    applyOutlier,
  ]);

  const handleSaveCleaned = async () => {
    if (!siteId || !fileName) {
      alert("缺少 site_id 或 file_name，請先回上一頁重新選擇");
      return;
    }

    const uploadId = localStorage.getItem("lastDataId"); // ⭐ 新增

    if (!uploadId) {
      alert("缺少 upload_id，請重新上傳資料");
      return;
    }

    setSaving(true);
    try {
      const body = {
        site_id: Number(siteId),
        file_name: fileName,
        upload_id: Number(uploadId), // ⭐ 新增這行

        apply_outlier: applyOutlier,
        apply_gi_tm: applyGiTm,
        remove_outliers: true,

        ...(applyOutlier && {
          outlier_method: outlierMethod,
          iqr_factor: iqrFactor,
          z_threshold: zThreshold,
          isolation_contamination: isolationContamination,
        }),
      };

      const res = await fetch("http://127.0.0.1:8000/save-cleaned-data/", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });

      if (!res.ok) {
        const text = await res.text();
        console.error("save-cleaned-data error:", text);
        throw new Error("儲存失敗");
      }

      const data = await res.json();

      if (data.after_id) {
        localStorage.setItem("afterDataId", data.after_id)
      }

      setResult({
        before_rows: data.before_rows,
        after_rows: data.after_rows,
        removed_ratio: data.removed_ratio,
      });
      setShowResultModal(true);

      onNext();
    } catch (err) {
      alert("儲存失敗，請稍後再試");
      console.error(err);
    } finally {
      setSaving(false);
    }
  };

  const tabs = [
    { key: "scatter", label: "散佈矩陣" },
    { key: "boxplot", label: "箱型圖" },
    { key: "correlation", label: "相關係數熱力圖" },
  ];

  const boxplotSubTabs = [
    { key: "month", label: "Month" },
    { key: "day", label: "Day" },
    { key: "hour", label: "Hour" },
  ];

  const renderContent = () => {
    if (!fileName || !siteId) {
      return (
        <div className="text-center py-20 text-white/60">
          缺少 site_id 或 file_name，請先回上一頁重新選擇資料
        </div>
      );
    }

    if (loading) return <div className="text-center py-20 text-white/60">資料載入中...</div>;
    if (!plots) return <div className="text-center py-20 text-white/60">無資料可顯示</div>;

    switch (selectedTab) {
        case "scatter":
          if (!plots.scatter_matrix) {
            return <div className="text-white/40">無散佈矩陣資料</div>;
          }
          return (
            <div ref={scatterRef}>
              <div className="grid grid-cols-3 gap-6">
                {plots.scatter_matrix.variables.map((v1) =>
                  plots.scatter_matrix.variables.map((v2) => {
                    if (v1 === v2) {
                      const hist = plots.scatter_matrix.hist?.[v1] || {
                        bins: [],
                        counts: [],
                      };

                      return (
                        <div key={`${v1}_${v2}`} style={{ width: 250 }}>
                          <HistogramSVG bins={hist.bins} counts={hist.counts} />
                        </div>
                      );
                    } else {
                      return (
                        <div key={`${v1}_${v2}`} style={{ width: 250 }}>
                          <RenderPairScatter
                            rowVar={v1}
                            colVar={v2}
                            plots={plots.scatter_matrix}
                          />
                        </div>
                      );
                    }
                  })
                )}
              </div>
            </div>
          );
      
        case "boxplot": {
          const rawHourKeys = stages?.raw?.boxplot_by_hour
            ? Object.keys(stages.raw.boxplot_by_hour)
                .map(Number)
                .sort((a, b) => a - b)
            : null;

          return (
            <div>
              <div ref={boxplotRef}>
                <div className="flex gap-4 justify-center mb-8">
                  {boxplotSubTabs.map((tab) => (
                    <button
                      key={tab.key}
                      onClick={() => setSelectedBoxplot(tab.key)}
                      className={`px-8 py-3 rounded-lg font-medium transition-all ${
                        selectedBoxplot === tab.key
                          ? "bg-blue-600 text-white shadow-lg"
                          : "bg-gray-800 text-white/70 hover:bg-gray-700"
                      }`}
                    >
                      {tab.label}
                    </button>
                  ))}
                </div>
                <div className="flex justify-center">
                  {selectedBoxplot === "month" && plots.boxplot_by_month && (
                    <BoxplotSVG groups={plots.boxplot_by_month} />
                  )}
                  {selectedBoxplot === "day" && plots.boxplot_by_day && (
                    <BoxplotSVG groups={plots.boxplot_by_day} />
                  )}
                  {selectedBoxplot === "hour" && plots.boxplot_by_hour && (
                    <BoxplotSVG
                      groups={plots.boxplot_by_hour}
                      expectedKeys={
                        rawHourKeys ||
                        Object.keys(plots.boxplot_by_hour)
                          .map(Number)
                          .sort((a, b) => a - b)
                      }
                    />
                  )}
                </div>
              </div>
            </div>
          );
        }
      
        case "correlation":
          const corrPlots = stages?.raw;
          return corrPlots?.correlation_heatmap_full ? (
            <div ref={corrRef}>
              <div className="flex justify-center">
                <CorrelationHeatmapSVG
                  variables={stages?.raw?.correlation_heatmap_full?.variables}
                  matrix={stages?.raw?.correlation_heatmap_full?.matrix}
                />
              </div>
            </div>
          ) : (
            <div className="text-white/40">無相關性熱圖資料</div>
          );
      default:
        return null;
    }
  };

  const [downloadOptions, setDownloadOptions] = useState({
    scatter: true,
    boxplot: true,
    correlation: true,
  });

  const handleDownloadPDF = async () => {
    await new Promise((resolve) => {
      setTimeout(() => {
        window.dispatchEvent(new Event("resize"));
        resolve();
      }, 500);
    });

    await document.fonts.ready;

    const pdf = new jsPDF("p", "mm", "a4");
    let y = 10;
    const pageHeight = 280;

    const addSection = async (ref, title) => {
      if (!ref.current) return;

      const canvas = await html2canvas(ref.current, {
        scale: 2, // ⭐ 提升清晰度
        useCORS: true,
        width: ref.current.scrollWidth,
        height: ref.current.scrollHeight,
      });

      const imgData = canvas.toDataURL("image/png");
      const imgWidth = 190;
      let imgHeight = (canvas.height * imgWidth) / canvas.width;

      const maxHeight = 260;
      if (imgHeight > maxHeight) {
        imgHeight = maxHeight;
      }

      // 換頁
      if (y + imgHeight > pageHeight) {
        pdf.addPage();
        y = 10;
      }

      pdf.setFontSize(14);
      pdf.text(title, 10, y);
      y += 6;

      pdf.addImage(imgData, "PNG", 10, y, imgWidth, imgHeight);
      y += imgHeight + 10;
    };

    // ⭐⭐ 重點：真的去加內容
    if (downloadOptions.scatter) {
      await addSection(pdfScatterRef, "Scatter Matrix");
    }

    if (downloadOptions.boxplot) {
      await addSection(pdfBoxplotRef, "Boxplot");
    }

    if (downloadOptions.correlation) {
      await addSection(pdfCorrRef, "Correlation Heatmap");
    }

    // ⭐⭐ 最重要：下載
    pdf.save("data_visualization.pdf");
  };

  return (
    <div className="min-h-screen bg-background-dark text-white">
      <Navbar
        activePage="data-cleaning"
        onNavigateToDashboard={onNavigateToDashboard}
        onNavigateToTrain={onNavigateToTrain}
        onNavigateToSites={onNavigateToSites}
        onNavigateToPredict={onNavigateToPredict}
        onNavigateToModelMgmt={onNavigateToModelMgmt}
        onLogout={onLogout}
      />

      <div className="w-full border-b border-white/10 bg-white/[.02] px-6 py-3 sticky top-[64px] sm:top-[65px] z-40 backdrop-blur-md">
        <div className="mx-auto flex max-w-7xl items-center justify-between">
          <button
            onClick={onBack}
            className="flex items-center gap-1 text-sm text-white/50 hover:text-white transition-colors"
          >
            <span className="material-symbols-outlined !text-lg">arrow_back</span>
            返回上一步
          </button>

          <div className="text-sm font-medium">
            <span className="text-white/40">1. 上傳資料</span>
            <span className="mx-2 text-white/30">/</span>
            <span className="text-primary font-bold">2. 清理資料</span>
            <span className="mx-2 text-white/30">/</span>
            <span className="text-white/40 ">3. 調整單位</span>
            <span className="mx-2 text-white/30">/</span>
            <span className="text-white/40">4. 模型訓練</span>
          </div>
        </div>
      </div>

      <main className="container mx-auto px-6 py-8 max-w-7xl">
        <h1 className="text-3xl font-bold mb-8 text-white">資料清理與視覺化</h1>
        <div className="flex flex-col gap-6 mb-8">
          <div className="flex gap-8 border-b border-white/10 pb-4 justify">
            {tabs.map((tab) => (
              <button
                key={tab.key}
                onClick={() => setSelectedTab(tab.key)}
                className={`text-lg font-semibold px-6 py-3 rounded-t-lg transition-all ${
                  selectedTab === tab.key
                    ? "bg-[#1E1E1E] text-blue-400 border-b-4 border-blue-400"
                    : "text-white/50 hover:text-white/80"
                }`}
              >
                {tab.label}
              </button>
            ))}
          </div>

          <div className="flex flex-wrap items-center gap-8 bg-black/20 p-4 rounded-xl">
            <label className="flex items-center gap-3 cursor-pointer">
              <input
                type="checkbox"
                checked={applyGiTm}
                onChange={(e) => {
                  setApplyGiTm(e.target.checked);
                  if (!e.target.checked) setApplyOutlier(false);
                }}
                className="w-5 h-5"
              />
              <span className="text-sm font-medium">GI = 0 刪除 / TM = 0 補值後</span>
            </label>

            <div className={`flex flex-wrap items-center gap-4 ${!applyGiTm && "opacity-40"}`}>
              <label className="flex items-center gap-3 cursor-pointer">
                <input
                  type="checkbox"
                  checked={applyOutlier}
                  disabled={!applyGiTm}
                  onChange={(e) => setApplyOutlier(e.target.checked)}
                  className="w-5 h-5"
                />
                <span className="text-sm font-medium">離群值處理</span>
              </label>

              <select
                value={outlierMethod}
                onChange={(e) => setOutlierMethod(e.target.value)}
                disabled={!applyOutlier}
                className="px-3 py-1 rounded bg-gray-800 text-sm disabled:opacity-40"
              >
                <option value="iqr_comprehensive">IQR（綜合）</option>
                <option value="iqr_single">IQR（單欄位）</option>
                <option value="zscore">Z-score</option>
                <option value="isolation_forest">Isolation Forest</option>
              </select>

              {outlierMethod.startsWith("iqr") && (
                <div className="flex items-center gap-2">
                  <input
                    type="number"
                    step="0.1"
                    value={iqrFactor}
                    disabled={!applyOutlier}
                    onChange={(e) => setIqrFactor(Number(e.target.value))}
                    className="w-20 bg-gray-800 px-2 py-1 rounded text-sm disabled:opacity-40"
                  />
                  <div className="text-xs text-white/60">
                    IQR 係數越小，判定越嚴格（建議 0.5–1.5）
                  </div>
                </div>
              )}

              {outlierMethod === "zscore" && (
                <div className="flex items-center gap-2">
                  <input
                    type="number"
                    step="0.1"
                    value={zThreshold}
                    disabled={!applyOutlier}
                    onChange={(e) => setZThreshold(Number(e.target.value))}
                    className="w-20 bg-gray-800 px-2 py-1 rounded text-sm disabled:opacity-40"
                  />
                  <div className="text-xs text-white/60">
                    Z 值越小，判定越嚴格（建議 2–3）
                  </div>
                </div>
              )}

              {outlierMethod === "isolation_forest" && (
                <div className="flex items-center gap-2">
                  <input
                    type="number"
                    step="0.01"
                    value={isolationContamination}
                    disabled={!applyOutlier}
                    onChange={(e) => setIsolationContamination(Number(e.target.value))}
                    className="w-20 bg-gray-800 px-2 py-1 rounded text-sm disabled:opacity-40"
                  />
                  <div className="text-xs text-white/60">
                    表示預期離群值比例（例如 0.05 ≈ 5%）
                  </div>
                </div>
              )}
            </div>
          </div>
        </div>

        <div className="bg-[#1E1E1E]/80 backdrop-blur rounded-2xl p-8 shadow-2xl">
          {renderContent()}
        </div>
      </main>

      <div className="sticky bottom-0 w-full border-t border-white/10 bg-background-dark/90 backdrop-blur-lg p-4 px-6 z-40">
        <div className="mx-auto flex max-w-6xl items-center justify-between">
          <div className="text-sm text-white/60">
            已根據檔案 {fileName} 產生視覺化{" "}
            {applyOutlier ? "（已套用離群值處理）" : "（紅色圓點僅標示離群值，未移除）"}
          </div>
          <div className="flex items-center gap-4">
            <div className="flex items-center gap-3 bg-black/20 px-4 py-2 rounded-xl">
              {/* 選項 */}
              <div className="flex items-center gap-4">

                <label className="flex items-center gap-2 text-sm cursor-pointer">
                  <input
                    type="checkbox"
                    checked={downloadOptions.scatter}
                    onChange={(e) =>
                      setDownloadOptions({
                        ...downloadOptions,
                        scatter: e.target.checked,
                      })
                    }
                    className="accent-blue-500"
                  />
                  散佈矩陣
                </label>

                <label className="flex items-center gap-2 text-sm cursor-pointer">
                  <input
                    type="checkbox"
                    checked={downloadOptions.boxplot}
                    onChange={(e) =>
                      setDownloadOptions({
                        ...downloadOptions,
                        boxplot: e.target.checked,
                      })
                    }
                    className="accent-blue-500"
                  />
                  箱型圖
                </label>

                <label className="flex items-center gap-2 text-sm cursor-pointer">
                  <input
                    type="checkbox"
                    checked={downloadOptions.correlation}
                    onChange={(e) =>
                      setDownloadOptions({
                        ...downloadOptions,
                        correlation: e.target.checked,
                      })
                    }
                    className="accent-blue-500"
                  />
                  相關熱圖
                </label>
              </div>

              {/* 分隔線 */}
              <div className="w-px h-6 bg-white/20" />

              {/* 下載按鈕 */}
              <button
                onClick={handleDownloadPDF}
                className="bg-green-500 hover:bg-green-400 transition px-4 py-2 rounded-lg text-sm font-bold text-black"
              >
                下載 PDF
              </button>
            </div>
            <button
              onClick={onBack}
              className="rounded-lg border border-white/10 px-6 py-2 text-sm font-bold text-white hover:bg-white/10"
            >
              返回
            </button>

            <button
              onClick={() => {
                const dataId = localStorage.getItem("lastDataId");

                if (!dataId) {
                  alert("沒有可訓練的資料，請先清理或重新上傳資料");
                  return;
                }

                onNext();
              }}
              className="rounded-lg border border-blue-400 px-6 py-2 text-sm font-bold text-blue-400 hover:bg-blue-400/10"
            >
              跳過清理 → 模型訓練
            </button>

            <button
              onClick={handleSaveCleaned}
              disabled={loading || saving || !plots || !applyOutlier || !siteId || !fileName}
              className="rounded-lg bg-primary px-8 py-2 text-sm font-bold text-background-dark disabled:opacity-50"
            >
              確認清理並繼續 → 模型訓練
            </button>
          </div>
        </div>
      </div>
      <div style={{ position: "fixed",
        top: 0,
        left: 0,
        opacity: 0,
        pointerEvents: "none",
        zIndex: -1}}>
  
        {downloadOptions.scatter && (
          <div ref={pdfScatterRef}style={{ width: "800px", background: "white", padding: "20px"}}>
            <h2>Scatter Matrix</h2>
            {plots?.scatter_matrix && (
              <div className="grid grid-cols-3 gap-6">
                {plots.scatter_matrix.variables.map((v1) =>
                  plots.scatter_matrix.variables.map((v2) => {
                    if (v1 === v2) {
                      const hist = plots.scatter_matrix.hist?.[v1] || {
                        bins: [],
                        counts: [],
                      };

                      return (
                        <div key={`${v1}_${v2}`} style={{ width: 250 }}>
                          <HistogramSVG bins={hist.bins} counts={hist.counts} />
                        </div>
                      );
                    } else {
                      return (
                        <div key={`${v1}_${v2}`} style={{ width: 250 }}>
                          <RenderPairScatter
                            rowVar={v1}
                            colVar={v2}
                            plots={plots.scatter_matrix}
                          />
                        </div>
                      );
                    }
                  })
                )}
              </div>
            )}
          </div>
        )}

        {downloadOptions.boxplot && (
          <div ref={pdfBoxplotRef}>
            <h2>Boxplot</h2>

            {plots?.boxplot_by_month && (
              <>
                <h3>Month</h3>
                <BoxplotSVG groups={plots.boxplot_by_month} />
              </>
            )}

            {plots?.boxplot_by_day && (
              <>
                <h3>Day</h3>
                <BoxplotSVG groups={plots.boxplot_by_day} />
              </>
            )}

            {plots?.boxplot_by_hour && (
              <>
                <h3>Hour</h3>
                <BoxplotSVG
                  groups={plots.boxplot_by_hour}
                  expectedKeys={rawHourKeys}
                />
              </>
            )}
          </div>
        )}

        {downloadOptions.correlation && (
          <div ref={pdfCorrRef}>
            <h2>Correlation</h2>
            {stages?.raw?.correlation_heatmap_full && (
              <CorrelationHeatmapSVG
                variables={stages.raw.correlation_heatmap_full.variables}
                matrix={stages.raw.correlation_heatmap_full.matrix}
              />
            )}
          </div>
        )}

      </div>
    </div>
  );
}