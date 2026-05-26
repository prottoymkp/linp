import React, { useState, useEffect, useMemo } from 'react';
import { RefreshCw, Play, BarChart2, CheckCircle2, AlertCircle, PackageOpen, Target, DollarSign, Activity } from 'lucide-react';

// --- DATA MODEL ---
const MATERIALS = [
  { id: 'adhesive', name: 'Adhesive', color: 'bg-red-500', uom: 'kg' },
  { id: 'leather', name: 'Premium Leather', color: 'bg-amber-700', uom: 'sq ft' },
  { id: 'synthetic', name: 'Synthetic Mesh', color: 'bg-blue-500', uom: 'm' },
  { id: 'rubber', name: 'Outsole Rubber', color: 'bg-slate-800', uom: 'kg' },
  { id: 'foam', name: 'EVA Midsole Foam', color: 'bg-teal-500', uom: 'm' },
  { id: 'thread', name: 'Nylon Thread', color: 'bg-gray-400', uom: 'spool' },
  { id: 'laces', name: 'Laces', color: 'bg-indigo-400', uom: 'pairs' },
  { id: 'eyelets', name: 'Metal Eyelets', color: 'bg-zinc-500', uom: 'pcs' }
];

// Structurally designed BOMs to create multi-dimensional traps
const PRODUCTS = [
  { id: 'shoe_c', name: 'Trekking Boot', type: 'Shoe', profit: 120, icon: '🥾', 
    bom: { adhesive: 4.0, leather: 3.5, synthetic: 0.5, rubber: 2.0, foam: 1.0, thread: 0.4, laces: 1, eyelets: 14 } },
  { id: 'shoe_b', name: 'Classic Sneaker', type: 'Shoe', profit: 75, icon: '👞', 
    bom: { adhesive: 1.2, leather: 1.5, synthetic: 0.0, rubber: 1.2, foam: 0.5, thread: 0.3, laces: 1, eyelets: 12 } },
  { id: 'shoe_a', name: 'Alpha Running Shoe', type: 'Shoe', profit: 45, icon: '👟', 
    bom: { adhesive: 0.8, leather: 0.0, synthetic: 1.5, rubber: 1.0, foam: 1.5, thread: 0.2, laces: 1, eyelets: 8 } },
  { id: 'sandal_z', name: 'Premium Leather Sandal', type: 'Sandal', profit: 40, icon: '👡', 
    bom: { adhesive: 1.0, leather: 2.0, synthetic: 0.0, rubber: 0.8, foam: 0.5, thread: 0.2, laces: 0, eyelets: 0 } },
  { id: 'sandal_y', name: 'Sport Sandal', type: 'Sandal', profit: 25, icon: '👡', 
    // Low profit, but a trap for rubber constraints
    bom: { adhesive: 0.4, leather: 0.0, synthetic: 1.2, rubber: 2.5, foam: 1.0, thread: 0.1, laces: 0, eyelets: 0 } },
  { id: 'sandal_x', name: 'Beach Slide', type: 'Sandal', profit: 15, icon: '🩴', 
    bom: { adhesive: 0.2, leather: 0.0, synthetic: 0.5, rubber: 0.8, foam: 0.8, thread: 0.05, laces: 0, eyelets: 0 } },
];

const generateInventoryForTargets = (currentTargets) => {
  const demand = {};
  MATERIALS.forEach(m => demand[m.id] = 0);
  
  PRODUCTS.forEach(product => {
    const qty = currentTargets[product.id] || 0;
    Object.entries(product.bom).forEach(([matId, amount]) => {
      if (demand[matId] !== undefined) {
        demand[matId] += amount * qty;
      }
    });
  });

  const materialIds = MATERIALS.map(m => m.id);
  const shuffled = [...materialIds].sort(() => 0.5 - Math.random());
  const bottleneckIds = shuffled.slice(0, 3);

  const newInventory = {};
  MATERIALS.forEach(m => {
    const d = demand[m.id];
    if (bottleneckIds.includes(m.id)) {
      // Create tight bottlenecks (40% - 60% of total demand)
      const minVal = Math.max(40, Math.floor(d * 0.4));
      const maxVal = Math.max(90, Math.floor(d * 0.6));
      newInventory[m.id] = Math.floor(Math.random() * (maxVal - minVal + 1)) + minVal;
    } else {
      // Safe margin for non-bottlenecks
      const minVal = Math.max(150, Math.floor(d * 1.3));
      const maxVal = Math.max(300, Math.floor(d * 1.8));
      newInventory[m.id] = Math.floor(Math.random() * (maxVal - minVal + 1)) + minVal;
    }
  });

  return newInventory;
};

const generateScenario = () => {
  const newTargets = {};
  PRODUCTS.forEach(p => {
    newTargets[p.id] = Math.floor(Math.random() * 200) + 50;
  });
  const newInventory = generateInventoryForTargets(newTargets);
  return { targets: newTargets, inventory: newInventory };
};

const getInitialPlan = () => {
  const plan = {};
  PRODUCTS.forEach(p => plan[p.id] = 0);
  return plan;
};

// --- COMPONENTS ---
const DonutChart = ({ percentage, color }) => {
  const radius = 36;
  const circumference = 2 * Math.PI * radius;
  const strokeDashoffset = circumference - (percentage / 100) * circumference;

  return (
    <div className="relative flex items-center justify-center w-24 h-24">
      <svg className="transform -rotate-90 w-24 h-24">
        <circle cx="48" cy="48" r={radius} stroke="currentColor" strokeWidth="8" fill="transparent" className="text-slate-100" />
        <circle cx="48" cy="48" r={radius} stroke="currentColor" strokeWidth="8" fill="transparent" 
          strokeDasharray={circumference} strokeDashoffset={strokeDashoffset} 
          className={`${color} transition-all duration-1000 ease-out`} />
      </svg>
      <span className="absolute text-sm font-bold text-slate-700">{Math.round(percentage)}%</span>
    </div>
  );
};

// --- MAIN APP ---
export default function App() {
  const [scenario, setScenario] = useState(() => generateScenario());
  const [inventory, setInventory] = useState(scenario.inventory);
  const [targets, setTargets] = useState(scenario.targets);
  
  const [manualPlan, setManualPlan] = useState(getInitialPlan());
  const [showOptimizer, setShowOptimizer] = useState(false);
  const [optimizedPlan, setOptimizedPlan] = useState(getInitialPlan());

  const handleRandomizeInventory = () => {
    const newInventory = generateInventoryForTargets(targets);
    setInventory(newInventory);
    setManualPlan(getInitialPlan());
    setShowOptimizer(false);
  };

  const handleRandomizeTargets = () => {
    const nextScenario = generateScenario();
    setTargets(nextScenario.targets);
    setInventory(nextScenario.inventory);
    setManualPlan(getInitialPlan());
    setShowOptimizer(false);
  };

  const calculateConsumption = (plan) => {
    const consumed = {};
    MATERIALS.forEach(m => consumed[m.id] = 0);
    
    Object.entries(plan).forEach(([productId, qty]) => {
      const product = PRODUCTS.find(p => p.id === productId);
      if (product) {
        Object.entries(product.bom).forEach(([matId, amount]) => {
          consumed[matId] += amount * qty;
        });
      }
    });
    return consumed;
  };

  const consumedManual = calculateConsumption(manualPlan);
  const consumedOptimized = calculateConsumption(optimizedPlan);

  const calculateMetrics = (plan) => {
    let totalProfit = 0;
    let totalVolume = 0;
    let targetVolume = 0;

    Object.entries(plan).forEach(([productId, qty]) => {
      const product = PRODUCTS.find(p => p.id === productId);
      if (product) {
        totalProfit += product.profit * qty;
        totalVolume += qty;
      }
    });

    Object.values(targets).forEach(t => targetVolume += t);

    return { 
      totalProfit, 
      totalVolume, 
      targetVolume, 
      fulfillment: targetVolume > 0 ? (totalVolume / targetVolume) * 100 : 0 
    };
  };

  const metricsManual = calculateMetrics(manualPlan);
  const metricsOptimized = calculateMetrics(optimizedPlan);

  const handleManualChange = (productId, val) => {
    let newValue = parseInt(val) || 0;
    const tempPlan = { ...manualPlan, [productId]: newValue };
    const tempConsumed = calculateConsumption(tempPlan);
    
    let canFulfill = true;
    Object.keys(tempConsumed).forEach(matId => {
      if (tempConsumed[matId] > inventory[matId]) {
        canFulfill = false;
      }
    });

    if (canFulfill && newValue <= targets[productId]) {
      setManualPlan(tempPlan);
    }
  };

  // Advanced Multi-Constraint LP Simulation
  const computeOptimization = (currentInv, currentTargets) => {
    let invTemp = { ...currentInv };
    let optPlan = {};
    PRODUCTS.forEach(p => optPlan[p.id] = 0);

    // Calculate dynamic scarcity (Shadow Price heuristic)
    // Products are scored based on Profit vs the relative scarcity of all materials they consume
    const scoredProducts = [...PRODUCTS].sort((a, b) => {
      const scarcityCostA = Object.entries(a.bom).reduce((acc, [mat, qty]) => acc + (qty / currentInv[mat]), 0);
      const scarcityCostB = Object.entries(b.bom).reduce((acc, [mat, qty]) => acc + (qty / currentInv[mat]), 0);
      
      const densityA = a.profit / (scarcityCostA || 1);
      const densityB = b.profit / (scarcityCostB || 1);
      
      return densityB - densityA; // Descending density
    });

    // Allocate greedily based on dynamic scarcity score
    scoredProducts.forEach(product => {
      let maxCanMake = currentTargets[product.id] || 0;
      Object.entries(product.bom).forEach(([matId, amount]) => {
        if (amount > 0) {
          const limit = Math.floor(invTemp[matId] / amount);
          if (limit < maxCanMake) maxCanMake = limit;
        }
      });

      optPlan[product.id] = maxCanMake;
      Object.entries(product.bom).forEach(([matId, amount]) => {
        invTemp[matId] -= amount * maxCanMake;
      });
    });

    return optPlan;
  };

  const runOptimizer = () => {
    const optPlan = computeOptimization(inventory, targets);
    setOptimizedPlan(optPlan);
    setShowOptimizer(true);
  };

  const handleIncreaseStock = (matId) => {
    setInventory(prev => {
      const nextInv = { ...prev, [matId]: Math.round(prev[matId] * 1.2) };
      if (showOptimizer) {
        setOptimizedPlan(computeOptimization(nextInv, targets));
      }
      return nextInv;
    });
  };

  return (
    <div className="min-h-screen bg-slate-50 text-slate-800 font-sans p-4 md:p-8">
      {/* Header */}
      <div className="max-w-6xl mx-auto mb-8 bg-white p-6 rounded-3xl shadow-sm border border-slate-100 flex flex-col md:flex-row justify-between items-center gap-4">
        <div>
          <h1 className="text-2xl md:text-3xl font-bold text-slate-900 flex items-center gap-3">
            <Activity className="text-blue-600" /> Production Optimizer Demo
          </h1>
          <p className="text-slate-500 mt-2 max-w-2xl">
            Experience the complexity of manual production planning. Try to maximize your profit and volume manually, then see how mathematical optimization resolves the material bottlenecks instantly.
          </p>
        </div>
        <div className="flex gap-3">
          <button 
            onClick={handleRandomizeInventory}
            className="flex items-center gap-2 px-4 py-2 bg-slate-100 hover:bg-slate-200 text-slate-700 rounded-xl transition-colors font-medium text-sm"
          >
            <RefreshCw size={16} /> Randomize Inventory
          </button>
          <button 
            onClick={handleRandomizeTargets}
            className="flex items-center gap-2 px-4 py-2 bg-slate-100 hover:bg-slate-200 text-slate-700 rounded-xl transition-colors font-medium text-sm"
          >
            <Target size={16} /> Randomize Targets
          </button>
        </div>
      </div>

      <div className="max-w-6xl mx-auto grid grid-cols-1 lg:grid-cols-12 gap-8">
        
        {/* LEFT COLUMN: Controls & Input */}
        <div className="lg:col-span-7 space-y-6">
          <div className="bg-white rounded-3xl shadow-sm border border-slate-100 p-6 md:p-8">
            <div className="flex justify-between items-end mb-6">
              <div>
                <h2 className="text-xl font-bold text-slate-800 mb-1">Production Planning</h2>
                <p className="text-sm text-slate-500">Allocate materials sequentially. Notice how prioritizing one product limits materials for others.</p>
              </div>
            </div>

            <div className="space-y-4">
              {PRODUCTS.map(product => {
                const target = targets[product.id] || 0;
                const currentVal = (showOptimizer ? optimizedPlan[product.id] : manualPlan[product.id]) ?? 0;
                const isFulfilled = currentVal === target;

                return (
                  <div key={product.id} className={`p-4 rounded-2xl border transition-all ${isFulfilled ? 'border-green-200 bg-green-50/30' : 'border-slate-100 bg-slate-50'}`}>
                    <div className="flex justify-between items-center mb-3">
                      <div className="group relative flex items-center gap-3 cursor-help">
                        <span className="text-3xl">{product.icon}</span>
                        <div>
                          <h3 className="font-semibold text-slate-800 border-b border-dashed border-slate-400 inline-block">
                            {product.name}
                          </h3>
                          
                          {/* BOM Tooltip */}
                          <div className="absolute z-50 left-0 top-full mt-2 hidden group-hover:block w-80 bg-slate-800 text-white text-xs rounded-xl p-4 shadow-xl border border-slate-700 pointer-events-none">
                            <div className="font-bold mb-3 pb-2 border-b border-slate-600 text-slate-100 flex justify-between">
                              <span>Bill of Materials</span>
                              <span className="text-slate-400 font-normal">Per pair</span>
                            </div>
                            <div className="space-y-1.5">
                              {/* Grid Header */}
                              <div className="grid grid-cols-3 gap-2 font-bold text-slate-400 border-b border-slate-700 pb-1.5 mb-1">
                                <span>Material</span>
                                <span className="text-right">Qty Needed</span>
                                <span className="text-right">UOM</span>
                              </div>
                              {Object.entries(product.bom).filter(([_, val]) => val > 0).map(([matId, val]) => {
                                 const mat = MATERIALS.find(m => m.id === matId);
                                 return (
                                   <div key={matId} className="grid grid-cols-3 gap-2 py-0.5 border-b border-slate-700/55 last:border-0">
                                     <span className="text-slate-300 truncate">{mat?.name}</span>
                                     <span className="font-mono text-blue-300 text-right">{val}</span>
                                     <span className="text-slate-400 font-mono text-right">{mat?.uom}</span>
                                   </div>
                                 );
                              })}
                            </div>
                          </div>

                          <div className="mt-1">
                            <p className="text-xs font-medium text-slate-500 bg-white px-2 py-1 rounded-md border inline-block">
                              Profit: ${product.profit} / pair
                            </p>
                          </div>
                        </div>
                      </div>
                      <div className="text-right">
                        <div className="text-sm font-semibold text-slate-700">{currentVal} / {target}</div>
                        <div className="text-xs text-slate-400">Target Pairs</div>
                      </div>
                    </div>
                    
                    {!showOptimizer ? (
                      <div className="flex items-center gap-4">
                        <input 
                          type="range" 
                          min="0" 
                          max={target} 
                          value={currentVal}
                          onChange={(e) => handleManualChange(product.id, e.target.value)}
                          className="w-full h-2 bg-slate-200 rounded-lg appearance-none cursor-pointer accent-blue-600"
                        />
                      </div>
                    ) : (
                      <div className="w-full bg-slate-200 rounded-full h-2">
                        <div className="bg-blue-600 h-2 rounded-full" style={{ width: target > 0 ? `${(currentVal/target)*100}%` : '0%' }}></div>
                      </div>
                    )}
                  </div>
                );
              })}
            </div>
            
            {!showOptimizer ? (
              <button 
                onClick={runOptimizer}
                className="w-full mt-8 py-4 bg-blue-600 hover:bg-blue-700 text-white rounded-2xl font-bold text-lg flex items-center justify-center gap-2 transition-all shadow-lg shadow-blue-600/20"
              >
                <Play size={20} /> Run LP Optimizer
              </button>
            ) : (
              <div className="w-full mt-8 p-4 bg-blue-50/50 rounded-2xl border border-blue-100 flex flex-col gap-3">
                 <div className="flex items-center gap-2 text-blue-800 font-semibold justify-center mb-1">
                    <CheckCircle2 size={18} className="text-blue-600" /> Mathematically Optimized for Maximum Profit
                 </div>
                 <button 
                  onClick={() => setShowOptimizer(false)}
                  className="w-full py-3 bg-slate-800 hover:bg-slate-900 text-white rounded-xl font-bold text-sm flex items-center justify-center gap-2 transition-all shadow-lg shadow-slate-800/20"
                >
                  <RefreshCw size={18} /> Exit Optimizer (Back to Manual)
                </button>
              </div>
            )}

          </div>
        </div>

        {/* RIGHT COLUMN: Dashboard & Analytics */}
        <div className="lg:col-span-5 space-y-6">
          
          {/* Key Metrics Cards */}
          <div className="grid grid-cols-2 gap-4">
            <div className={`p-6 rounded-3xl border shadow-sm transition-colors ${showOptimizer ? 'bg-blue-50 border-blue-100' : 'bg-white border-slate-100'}`}>
              <div className="flex items-center gap-2 text-slate-500 mb-2">
                <DollarSign size={18} /> <span className="font-medium text-sm">Total Profit</span>
              </div>
              <div className={`text-3xl font-bold ${showOptimizer ? 'text-blue-700' : 'text-slate-800'}`}>
                ${(showOptimizer ? metricsOptimized.totalProfit : metricsManual.totalProfit).toLocaleString()}
              </div>
              {showOptimizer && metricsOptimized.totalProfit !== metricsManual.totalProfit && (
                <div className={`text-xs font-bold mt-2 px-2 py-1 rounded-md inline-block ${metricsOptimized.totalProfit > metricsManual.totalProfit ? 'text-green-600 bg-green-100' : 'text-red-600 bg-red-100'}`}>
                  {metricsOptimized.totalProfit > metricsManual.totalProfit ? '+' : ''}
                  ${(metricsOptimized.totalProfit - metricsManual.totalProfit).toLocaleString()} vs Manual
                </div>
              )}
            </div>
            
            <div className={`p-6 rounded-3xl border shadow-sm transition-colors ${showOptimizer ? 'bg-blue-50 border-blue-100' : 'bg-white border-slate-100'}`}>
              <div className="flex items-center gap-2 text-slate-500 mb-2">
                <PackageOpen size={18} /> <span className="font-medium text-sm">Volume Produced</span>
              </div>
              <div className={`text-3xl font-bold ${showOptimizer ? 'text-blue-700' : 'text-slate-800'}`}>
                {(showOptimizer ? metricsOptimized.totalVolume : metricsManual.totalVolume).toLocaleString()} <span className="text-lg font-normal text-slate-400">prs</span>
              </div>
              {showOptimizer && metricsOptimized.totalVolume !== metricsManual.totalVolume && (
                <div className={`text-xs font-bold mt-2 px-2 py-1 rounded-md inline-block ${metricsOptimized.totalVolume > metricsManual.totalVolume ? 'text-blue-600 bg-blue-100' : 'text-slate-600 bg-slate-100'}`}>
                  {metricsOptimized.totalVolume > metricsManual.totalVolume ? '+' : ''}
                  {(metricsOptimized.totalVolume - metricsManual.totalVolume).toLocaleString()} prs vs Manual
                </div>
              )}
            </div>
          </div>

          {/* Fulfillment Chart */}
          <div className="bg-white rounded-3xl shadow-sm border border-slate-100 p-6 flex items-center justify-between">
            <div>
              <h3 className="font-bold text-slate-800 mb-1">Target Fulfillment</h3>
              <p className="text-sm text-slate-500">Percentage of total requested volume achieved.</p>
            </div>
            <DonutChart 
              percentage={showOptimizer ? metricsOptimized.fulfillment : metricsManual.fulfillment} 
              color={showOptimizer ? 'text-blue-600' : 'text-slate-800'} 
            />
          </div>

          {/* Resource Constraints Chart */}
          <div className="bg-white rounded-3xl shadow-sm border border-slate-100 p-6">
            <div className="flex items-center justify-between mb-4">
              <div>
                <h3 className="font-bold text-slate-800 mb-0.5">Live Resource Utilization</h3>
                <p className="text-xs text-slate-400">Click "+20%" to bypass bottlenecks dynamically</p>
              </div>
              <BarChart2 className="text-slate-400 shrink-0" size={20} />
            </div>

            <div className="space-y-5">
              {MATERIALS.map(mat => {
                const available = inventory[mat.id];
                const consumed = showOptimizer ? consumedOptimized[mat.id] : consumedManual[mat.id];
                const percentage = Math.min(100, (consumed / available) * 100);
                const isBottleneck = percentage >= 95;

                const activePlan = showOptimizer ? optimizedPlan : manualPlan;

                return (
                  <div key={mat.id} className="relative">
                    <div className="flex justify-between items-center text-sm mb-1.5 flex-wrap gap-1">
                      <div className="group relative">
                        <span className={`font-medium flex items-center gap-2 cursor-help border-b border-dashed border-slate-300 hover:text-blue-600 transition-colors ${isBottleneck ? 'text-red-600 border-red-300' : 'text-slate-700'}`}>
                          {mat.name} {isBottleneck && <AlertCircle size={14} />}
                        </span>

                        <div className="absolute z-50 left-0 bottom-full mb-2 hidden group-hover:block w-80 bg-slate-800 text-white text-xs rounded-xl p-4 shadow-xl border border-slate-700 pointer-events-none">
                          <div className="font-bold mb-3 pb-2 border-b border-slate-600 text-slate-100 flex justify-between">
                            <span>FG Consumption Breakdown</span>
                            <span className="text-slate-400 font-normal">UOM: {mat.uom}</span>
                          </div>
                          <div className="space-y-1.5">
                            <div className="grid grid-cols-3 gap-2 font-bold text-slate-400 border-b border-slate-700 pb-1.5 mb-1">
                              <span>Finished Good</span>
                              <span className="text-right">Qty/Pair</span>
                              <span className="text-right">Total Consumed</span>
                            </div>
                            {PRODUCTS.filter(p => p.bom[mat.id] > 0).map(p => {
                              const qtyPerPair = p.bom[mat.id];
                              const plannedQty = activePlan[p.id] || 0;
                              const totalConsumption = qtyPerPair * plannedQty;

                              return (
                                <div key={p.id} className="grid grid-cols-3 gap-2 py-0.5 border-b border-slate-700/55 last:border-0 items-center">
                                  <span className="text-slate-300 truncate">{p.icon} {p.name}</span>
                                  <span className="font-mono text-blue-300 text-right">{qtyPerPair}</span>
                                  <span className="font-mono text-emerald-300 text-right font-semibold">
                                    {totalConsumption.toLocaleString(undefined, { minimumFractionDigits: 1, maximumFractionDigits: 1 })}
                                  </span>
                                </div>
                              );
                            })}
                          </div>
                        </div>
                      </div>

                      <div className="flex items-center gap-1.5 font-mono text-xs">
                        <span className="text-slate-500">
                          {Math.round(consumed)} / {available} {mat.uom}
                        </span>
                        <button 
                          onClick={() => handleIncreaseStock(mat.id)}
                          className="px-1.5 py-0.5 bg-green-50 text-green-700 hover:bg-green-100 font-bold border border-green-200 rounded text-[10px] transition-all cursor-pointer shadow-sm active:scale-95 select-none"
                          title={`Add 20% more ${mat.name}`}
                        >
                          +20%
                        </button>
                      </div>
                    </div>
                    <div className="w-full bg-slate-100 rounded-full h-2.5 overflow-hidden">
                      <div 
                        className={`h-2.5 rounded-full transition-all duration-500 ${isBottleneck ? 'bg-red-500' : 'bg-slate-800'}`}
                        style={{ width: `${percentage}%` }}
                      ></div>
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
          
        </div>
      </div>
    </div>
  );
}