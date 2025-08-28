import React, { useEffect, useRef, useState } from "react";
import Globe from "react-globe.gl";

// Helper: stratified sampling
function stratifiedSample(points, targetSize = 8000) {
  if (points.length <= targetSize) return points;
  const latBins = 20, lonBins = 30;
  const sampled = [];
  const groups = {};

  points.forEach(p => {
    const latBin = Math.floor(((p.lat + 90) / 180) * latBins);
    const lonBin = Math.floor(((p.lng + 180) / 360) * lonBins);
    const key = `${latBin}-${lonBin}`;
    if (!groups[key]) groups[key] = [];
    groups[key].push(p);
  });

  const perBin = Math.floor(targetSize / Object.keys(groups).length);
  Object.values(groups).forEach(binPoints => {
    for (let i = 0; i < Math.min(perBin, binPoints.length); i++) {
      const idx = Math.floor(Math.random() * binPoints.length);
      sampled.push(binPoints[idx]);
    }
  });

  return sampled;
}

export default function AtmosphericGlobe({ apiBase = "" }) {
  const globeEl = useRef();
  const [points, setPoints] = useState([]);
  const [hovered, setHovered] = useState(null);

  // Layer toggles
  const [showWind, setShowWind] = useState(true);
  const [showCloud, setShowCloud] = useState(true);
  const [showAerosol, setShowAerosol] = useState(true);
  const [showDroplet, setShowDroplet] = useState(true);
  const [showBoundary, setShowBoundary] = useState(true);

  const [pointScale, setPointScale] = useState(1.2);

  // Mock loader (replace with API fetch if needed)
  useEffect(() => {
    const n = 700000;
    const raw = Array.from({ length: n }).map(() => {
      const lat = (Math.random() - 0.5) * 180;
      const lng = (Math.random() - 0.5) * 360;
      const wind = Math.random() * 12;
      const cloud = Math.random();
      const aerosol = Math.random() * 150;
      const droplet = 50 + Math.random() * 100;
      const boundary = Math.random() * 1200;
      const score = 0.25 * (cloud) + 0.25 * (wind / 12) + 
                    0.2 * (boundary / 1200) + 0.15 * (droplet / 150) +
                    0.15 * (aerosol / 150);

      let condition = "Low";
      if (score >= 0.66) condition = "High";
      else if (score >= 0.33) condition = "Medium";

      return { lat, lng, wind, cloud, aerosol, droplet, boundary, score, condition };
    });

    setPoints(stratifiedSample(raw, 8000));
  }, []);

  // Softer teal palette
  const colorMap = {
    High: "rgba(13,115,119,0.8)",   // Deep teal
    Medium: "rgba(20,160,133,0.7)", // Bright cyan
    Low: "rgba(127,205,205,0.6)"    // Soft aqua
  };

  const radiusMap = {
    High: 0.6,
    Medium: 0.4,
    Low: 0.25
  };

  // Apply layer filters (only show points if corresponding layers are ON)
  const filteredPoints = points.filter(p => {
    return (
      (showWind && p.wind > 0) ||
      (showCloud && p.cloud > 0) ||
      (showAerosol && p.aerosol > 0) ||
      (showDroplet && p.droplet > 0) ||
      (showBoundary && p.boundary > 0)
    );
  });

  return (
    <div className="min-h-screen bg-[#041323] text-white font-sans">
      <div className="max-w-7xl mx-auto p-4">
        <header className="flex items-center justify-between mb-4">
          <div>
            <h1 className="text-2xl font-semibold">Atmospheric Conditions Globe</h1>
            <p className="text-sm text-slate-300">
              Toggle layers, hover points for details, explore atmospheric KPIs globally
            </p>
          </div>
        </header>

        <div className="grid grid-cols-12 gap-4">
          {/* Globe */}
          <div className="col-span-9 relative rounded-xl overflow-hidden shadow-2xl" style={{ height: 680 }}>
            <Globe
              ref={globeEl}
              globeImageUrl="https://unpkg.com/three-globe/example/img/earth-night.jpg"
              backgroundColor="#041323"
              pointsData={filteredPoints}
              pointLat={d => d.lat}
              pointLng={d => d.lng}
              pointAltitude={d => 0.01 + radiusMap[d.condition] * 0.002}
              pointColor={d => colorMap[d.condition]}
              pointRadius={d => radiusMap[d.condition] * pointScale}
              onPointHover={setHovered}
              atmosphereColor={"#0ea5a5"}
              showAtmosphere={true}
              atmosphereRadiusScale={1.05}
              animateIn={true}
            />

            {/* Hover tooltip */}
            {hovered && (
              <div className="absolute left-4 top-4 bg-[#061425]/80 backdrop-blur-md p-3 rounded-lg border border-white/5 z-40 max-w-xs">
                <div className="text-sm font-semibold">Condition: {hovered.condition}</div>
                <div className="text-xs text-slate-300">
                  Lat: {hovered.lat.toFixed(2)} · Lon: {hovered.lng.toFixed(2)}
                </div>
                <div className="mt-2 text-xs">
                  <div>Score: {hovered.score.toFixed(2)}</div>
                </div>
              </div>
            )}

            {/* Controls overlay */}
            <div className="absolute right-4 bottom-4 bg-[#061425]/70 p-3 rounded-lg border border-white/5 text-xs">
              <div className="font-medium mb-2">Layers</div>
              <div className="flex flex-col gap-1">
                <label><input type="checkbox" checked={showWind} onChange={() => setShowWind(!showWind)} /> Wind Speed</label>
                <label><input type="checkbox" checked={showCloud} onChange={() => setShowCloud(!showCloud)} /> Cloud Coverage</label>
                <label><input type="checkbox" checked={showAerosol} onChange={() => setShowAerosol(!showAerosol)} /> Aerosol</label>
                <label><input type="checkbox" checked={showDroplet} onChange={() => setShowDroplet(!showDroplet)} /> Droplet Conc.</label>
                <label><input type="checkbox" checked={showBoundary} onChange={() => setShowBoundary(!showBoundary)} /> Boundary Height</label>
                <label className="flex gap-2 items-center mt-2">
                  Point Scale
                  <input
                    type="range"
                    min="0.5" max="2" step="0.1"
                    value={pointScale}
                    onChange={e => setPointScale(parseFloat(e.target.value))}
                  />
                </label>
              </div>
            </div>
          </div>

          {/* Side Panel */}
          <aside className="col-span-3 space-y-4">
            <div className="bg-white/5 p-4 rounded-2xl shadow-md h-72 overflow-auto">
              <h3 className="font-semibold">Selected Site</h3>
              {!hovered ? (
                <div className="text-sm text-slate-300 mt-3">Hover over a point to see details</div>
              ) : (
                <div className="mt-3 text-sm space-y-2">
                  <div className="font-medium">Condition: {hovered.condition}</div>
                  <div className="text-xs text-slate-300">
                    Lat: {hovered.lat.toFixed(2)} · Lon: {hovered.lng.toFixed(2)}
                  </div>
                  <div>Score: {hovered.score.toFixed(2)}</div>
                  <div>Wind: {hovered.wind.toFixed(1)} m/s</div>
                  <div>Cloud Coverage: {hovered.cloud.toFixed(2)}</div>
                  <div>Aerosol: {hovered.aerosol.toFixed(1)} µg/m³</div>
                  <div>Droplet Conc.: {hovered.droplet.toFixed(1)} cm⁻³</div>
                  <div>Boundary Height: {hovered.boundary.toFixed(0)} m</div>
                  <div className="pt-3">
                    <button
                      onClick={() =>
                        globeEl.current.pointOfView({ lat: hovered.lat, lng: hovered.lng, altitude: 1.5 }, 1200)
                      }
                      className="px-3 py-1 rounded bg-[#0D7377] text-black font-medium"
                    >
                      Fly to site
                    </button>
                  </div>
                </div>
              )}
            </div>
          </aside>
        </div>
      </div>
    </div>
  );
}
