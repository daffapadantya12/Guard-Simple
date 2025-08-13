import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { Shield, BarChart3, Settings, AlertTriangle, CheckCircle, XCircle, Database, Trash2, RefreshCw, Clock } from 'lucide-react';
import './App.css';

// API base URL - point to backend server
const API_BASE = process.env.NODE_ENV === 'development' ? 'http://localhost:8000' : (process.env.REACT_APP_API_URL || 'http://localhost:8000');

// API Key for authentication
const API_KEY = process.env.REACT_APP_API_KEY || 'dffhnfpdnty0194392429340';

// API client with authentication
const apiClient = axios.create({
  baseURL: API_BASE,
  timeout: 30000,
  headers: {
    'Content-Type': 'application/json',
    'Authorization': `Bearer ${API_KEY}`
  }
});



// Tab component
const TabButton = ({ active, onClick, icon: Icon, children }) => (
  <button
    onClick={onClick}
    className={`flex items-center px-4 py-2 text-sm font-medium rounded-md transition-colors ${
      active
        ? 'bg-blue-100 text-blue-700 border border-blue-200'
        : 'text-gray-600 hover:text-gray-900 hover:bg-gray-100'
    }`}
  >
    <Icon className="w-4 h-4 mr-2" />
    {children}
  </button>
);

// Decision badge component
const DecisionBadge = ({ decision }) => {
  const configs = {
    allow: { icon: CheckCircle, color: 'bg-green-100 text-green-800', text: 'SAFE' },
    warn: { icon: AlertTriangle, color: 'bg-yellow-100 text-yellow-800', text: 'WARNING' },
    block: { icon: XCircle, color: 'bg-red-100 text-red-800', text: 'BLOCKED' }
  };
  
  const config = configs[decision] || configs.warn;
  const Icon = config.icon;
  
  return (
    <div className={`inline-flex items-center px-3 py-1 rounded-full text-sm font-medium ${config.color}`}>
      <Icon className="w-4 h-4 mr-1" />
      {config.text}
    </div>
  );
};

// Guard result component
const GuardResult = ({ name, result, expanded, onToggle }) => {
  const getVerdictColor = (verdict) => {
    switch (verdict) {
      case 'allow': return 'text-green-600';
      case 'warn': return 'text-yellow-600';
      case 'block': return 'text-red-600';
      default: return 'text-gray-600';
    }
  };
  
  return (
    <div className="border rounded-lg p-4">
      <div 
        className="flex justify-between items-center cursor-pointer"
        onClick={onToggle}
      >
        <div className="flex items-center space-x-3">
          <span className="font-medium">{name.replace('_', ' ').toUpperCase()}</span>
          <span className={`font-semibold ${getVerdictColor(result.verdict)}`}>
            {result.verdict.toUpperCase()}
          </span>
        </div>
        <span className="text-gray-400">{expanded ? '−' : '+'}</span>
      </div>
      
      {expanded && (
        <div className="mt-3 pt-3 border-t space-y-2">
          {result.labels && result.labels.length > 0 && (
            <div>
              <span className="text-sm font-medium text-gray-600">Categories: </span>
              <span className="text-sm">{result.labels.join(', ')}</span>
            </div>
          )}
          {result.score !== null && result.score !== undefined && (
            <div>
              <span className="text-sm font-medium text-gray-600">Score: </span>
              <span className="text-sm">{result.score.toFixed(3)}</span>
            </div>
          )}
        </div>
      )}
    </div>
  );
};

// Input & Analysis Tab
const AnalysisTab = ({ apiClient }) => {
  const [prompt, setPrompt] = useState('');
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [expandedGuards, setExpandedGuards] = useState({});
  const [language, setLanguage] = useState('auto');
  
  const analyzePrompt = async () => {
    if (!prompt.trim() || !apiClient) return;
    
    setLoading(true);
    try {
      const response = await apiClient.post('/analyze', {
        prompt: prompt,
        lang: language
      });
      setResult(response.data);
    } catch (error) {
      console.error('Analysis failed:', error);
      if (error.response?.status === 403) {
        setResult({
          final: 'warn',
          per_guard: {
            error: { verdict: 'warn', labels: ['authentication_error'], score: null }
          },
          policy: { rule: 'Authentication failed - please check your API key' }
        });
      } else {
        setResult({
          final: 'warn',
          per_guard: {
            error: { verdict: 'warn', labels: ['system_error'], score: null }
          },
          policy: { rule: 'An error occurred' }
        });
      }
    }
    setLoading(false);
  };
  
  const toggleGuard = (guardName) => {
    setExpandedGuards(prev => ({
      ...prev,
      [guardName]: !prev[guardName]
    }));
  };
  
  return (
    <div className="space-y-6">
      <div className="bg-white rounded-lg shadow p-6">
        <h2 className="text-xl font-semibold mb-4">Prompt Analysis</h2>
        
        <div className="space-y-4">
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Language
            </label>
            <select 
              value={language} 
              onChange={(e) => setLanguage(e.target.value)}
              className="border border-gray-300 rounded-md px-3 py-2"
            >
              <option value="auto">Auto-detect</option>
              <option value="en">English</option>
              <option value="id">Indonesian</option>
            </select>
          </div>
          
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Enter your prompt
            </label>
            <textarea
              value={prompt}
              onChange={(e) => setPrompt(e.target.value)}
              placeholder="Type your prompt here..."
              className="w-full h-32 border border-gray-300 rounded-md px-3 py-2 focus:ring-2 focus:ring-blue-500 focus:border-transparent"
            />
          </div>
          
          <button
            onClick={analyzePrompt}
            disabled={loading || !prompt.trim()}
            className="bg-blue-600 text-white px-6 py-2 rounded-md hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed"
          >
            {loading ? 'Analyzing...' : 'Analyze Prompt'}
          </button>
        </div>
      </div>
      
      {result && (
        <div className="bg-white rounded-lg shadow p-6">
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-lg font-semibold">Analysis Result</h3>
            <DecisionBadge decision={result.final} />
          </div>
          
          {result.final === 'block' && (
            <div className="bg-red-50 border border-red-200 rounded-md p-4 mb-4">
              <p className="text-red-800">Prompt blocked for safety (e.g., violence, privacy risk).</p>
            </div>
          )}
          
          {result.final === 'warn' && (
            <div className="bg-yellow-50 border border-yellow-200 rounded-md p-4 mb-4">
              <p className="text-yellow-800">Prompt may violate policy—proceed with caution.</p>
            </div>
          )}
          
          <div className="space-y-3">
            <h4 className="font-medium text-gray-900">Per-Guard Breakdown</h4>
            {Object.entries(result.per_guard).map(([guardName, guardResult]) => (
              <GuardResult
                key={guardName}
                name={guardName}
                result={guardResult}
                expanded={expandedGuards[guardName]}
                onToggle={() => toggleGuard(guardName)}
              />
            ))}
          </div>
          
          <div className="mt-4 pt-4 border-t">
            <p className="text-sm text-gray-600">
              <strong>Policy:</strong> {result.policy.rule}
            </p>
          </div>
        </div>
      )}
    </div>
  );
};

// Dashboard Tab
const DashboardTab = ({ apiClient }) => {
  const [metrics, setMetrics] = useState(null);
  const [loading, setLoading] = useState(true);
  
  useEffect(() => {
    if (!apiClient) {
      setLoading(false);
      return;
    }

    const fetchMetrics = async () => {
      try {
        const response = await apiClient.get('/metrics');
        setMetrics(response.data);
      } catch (error) {
        console.error('Failed to fetch metrics:', error);
        if (error.response?.status === 403) {
          console.error('Authentication failed - API key may be invalid');
        }
      }
      setLoading(false);
    };
    
    fetchMetrics();
    const interval = setInterval(fetchMetrics, 30000); // Refresh every 30s
    return () => clearInterval(interval);
  }, [apiClient]);
  
  if (loading) {
    return <div className="text-center py-8">Loading metrics...</div>;
  }
  
  if (!metrics) {
    return <div className="text-center py-8 text-red-600">Failed to load metrics</div>;
  }
  
  const kpis = [
    {
      title: 'Total Prompts',
      value: metrics.overview?.total_requests || 0,
      color: 'bg-blue-500',
      icon: Shield
    },
    {
      title: '% Blocked',
      value: `${metrics.decisions?.percentages?.block_pct?.toFixed(1) || 0}%`,
      color: 'bg-red-500',
      icon: XCircle
    },
    {
      title: '% Warnings',
      value: `${metrics.decisions?.percentages?.warn_pct?.toFixed(1) || 0}%`,
      color: 'bg-yellow-500',
      icon: AlertTriangle
    },
    {
      title: 'Avg Latency',
      value: `${metrics.overview?.avg_latency_ms?.toFixed(0) || 0}ms`,
      color: 'bg-green-500',
      icon: Clock
    }
  ];
  
  return (
    <div className="space-y-6">
      {/* KPI Tiles */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        {kpis.map((kpi, index) => {
          const Icon = kpi.icon;
          return (
            <div key={index} className="bg-white rounded-lg shadow p-6">
              <div className="flex items-center">
                <div className={`w-12 h-12 rounded-lg ${kpi.color} flex items-center justify-center mr-4`}>
                  <Icon className="w-6 h-6 text-white" />
                </div>
                <div>
                  <p className="text-sm font-medium text-gray-600">{kpi.title}</p>
                  <p className="text-2xl font-bold text-gray-900">{kpi.value}</p>
                </div>
              </div>
            </div>
          );
        })}
      </div>
      
      {/* Decision Counts Chart */}
      <div className="bg-white rounded-lg shadow p-6">
        <h3 className="text-lg font-semibold mb-4">Decision Distribution</h3>
        <div className="space-y-3">
          {Object.entries(metrics.decisions?.counts || {}).map(([decision, count]) => {
            const total = metrics.overview?.total_requests || 1;
            const percentage = (count / total) * 100;
            const colors = {
              allow: 'bg-green-500',
              warn: 'bg-yellow-500',
              block: 'bg-red-500'
            };
            
            return (
              <div key={decision} className="flex items-center space-x-3">
                <div className="w-20 text-sm font-medium capitalize">{decision}</div>
                <div className="flex-1 bg-gray-200 rounded-full h-4">
                  <div 
                    className={`h-4 rounded-full ${colors[decision]}`}
                    style={{ width: `${percentage}%` }}
                  ></div>
                </div>
                <div className="w-16 text-sm text-gray-600">
                  {count} ({percentage.toFixed(1)}%)
                </div>
              </div>
            );
          })}
        </div>
      </div>
      
      {/* Blocking Reason Analysis */}
      <BlockingReasonAnalysis metrics={metrics} />
      
      {/* Cache Analysis */}
      <CacheAnalysisSection apiClient={apiClient} />
      
      {/* Guard Performance Table */}
      <div className="bg-white rounded-lg shadow p-6">
        <h3 className="text-lg font-semibold mb-4">Guard Performance</h3>
        <div className="overflow-x-auto">
          <table className="min-w-full divide-y divide-gray-200">
            <thead className="bg-gray-50">
              <tr>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Guard
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Requests
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Avg Latency
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Block Rate
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Error Rate
                </th>
              </tr>
            </thead>
            <tbody className="bg-white divide-y divide-gray-200">
              {Object.entries(metrics.per_guard_stats || {}).map(([guardName, stats]) => {
                const total = stats.total || 1;
                const blockRate = ((stats.block || 0) / total * 100).toFixed(1);
                const errorRate = ((stats.errors || 0) / total * 100).toFixed(1);
                
                return (
                  <tr key={guardName}>
                    <td className="px-6 py-4 whitespace-nowrap text-sm font-medium text-gray-900">
                      {guardName.replace('_', ' ').toUpperCase()}
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                      {stats.total || 0}
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                      {((stats.avg_latency || 0) * 1000).toFixed(0)}ms
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                      {blockRate}%
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                      <span className={errorRate > 5 ? 'text-red-600' : 'text-green-600'}>
                        {errorRate}%
                      </span>
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
};

// Blocking Reason Analysis Component
const BlockingReasonAnalysis = ({ metrics }) => {
  const categoryData = metrics?.categories?.frequent_triggers || {};
  const totalBlocked = metrics?.decisions?.counts?.block || 0;
  
  // Filter out system errors and get meaningful categories
  const meaningfulCategories = Object.entries(categoryData)
    .filter(([category, count]) => 
      category !== 'system_error' && 
      count > 0 && 
      !category.includes('_error')
    )
    .sort(([,a], [,b]) => b - a)
    .slice(0, 8); // Top 8 categories
  
  // Category display names and colors
  const getCategoryDisplay = (category) => {
    const displays = {
      'Violence': { name: 'Violence & Threats', color: 'bg-red-500', icon: '⚔️' },
      'Sexual_Minors': { name: 'Sexual Content (Minors)', color: 'bg-red-600', icon: '🚫' },
      'Cybercrime': { name: 'Cybercrime & Hacking', color: 'bg-purple-500', icon: '💻' },
      'Privacy': { name: 'Privacy Violation', color: 'bg-blue-500', icon: '🔒' },
      'Hate': { name: 'Hate Speech', color: 'bg-orange-500', icon: '💢' },
      'Self_Harm': { name: 'Self Harm', color: 'bg-pink-500', icon: '⚠️' },
      'high_toxicity': { name: 'High Toxicity', color: 'bg-red-400', icon: '☠️' },
      'moderate_toxicity': { name: 'Moderate Toxicity', color: 'bg-yellow-500', icon: '⚡' },
      'indonesian_toxicity': { name: 'Indonesian Toxicity', color: 'bg-red-300', icon: '🇮🇩' },
      'english_toxicity': { name: 'English Toxicity', color: 'bg-red-300', icon: '🇺🇸' },
      'LLMGuard_High_Risk': { name: 'LLM Guard High Risk', color: 'bg-gray-600', icon: '🛡️' },
      'Minors_Context': { name: 'Minors Context', color: 'bg-yellow-400', icon: '👶' },
      'Code_Generation': { name: 'Code Generation', color: 'bg-green-500', icon: '💾' },
      'Data_Leakage': { name: 'Data Leakage', color: 'bg-indigo-500', icon: '📊' }
    };
    
    return displays[category] || { 
      name: category.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase()), 
      color: 'bg-gray-400', 
      icon: '📋' 
    };
  };
  
  if (meaningfulCategories.length === 0) {
    return (
      <div className="bg-white rounded-lg shadow p-6">
        <h3 className="text-lg font-semibold mb-4 flex items-center">
          <AlertTriangle className="w-5 h-5 mr-2" />
          Blocking Reason Analysis
        </h3>
        <div className="text-center py-8 text-gray-500">
          <AlertTriangle className="w-12 h-12 mx-auto mb-3 text-gray-300" />
          <p>No blocking reasons recorded yet</p>
          <p className="text-sm">Submit some prompts to see blocking analysis</p>
        </div>
      </div>
    );
  }
  
  return (
    <div className="bg-white rounded-lg shadow p-6">
      <div className="flex justify-between items-center mb-4">
        <h3 className="text-lg font-semibold flex items-center">
          <AlertTriangle className="w-5 h-5 mr-2" />
          Blocking Reason Analysis
        </h3>
        <div className="text-sm text-gray-500">
          {totalBlocked} total blocks
        </div>
      </div>
      
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-6">
        {meaningfulCategories.map(([category, count]) => {
          const display = getCategoryDisplay(category);
          const percentage = totalBlocked > 0 ? ((count / totalBlocked) * 100).toFixed(1) : 0;
          
          return (
            <div key={category} className="bg-gray-50 rounded-lg p-4">
              <div className="flex items-center justify-between mb-2">
                <div className="flex items-center">
                  <span className="text-lg mr-2">{display.icon}</span>
                  <span className="font-medium text-gray-900">{display.name}</span>
                </div>
                <div className="text-right">
                  <div className="text-lg font-bold text-gray-900">{count}</div>
                  <div className="text-xs text-gray-500">{percentage}%</div>
                </div>
              </div>
              <div className="w-full bg-gray-200 rounded-full h-2">
                <div 
                  className={`h-2 rounded-full ${display.color} transition-all duration-300`}
                  style={{ width: `${Math.min(percentage, 100)}%` }}
                ></div>
              </div>
            </div>
          );
        })}
      </div>
      
      <div className="bg-blue-50 p-4 rounded-lg">
        <h4 className="font-medium mb-2 text-blue-900">📊 Insights & Recommendations</h4>
        <ul className="text-sm text-blue-800 space-y-1">
          {meaningfulCategories.length > 0 && (
            <li>• Most common blocking reason: <strong>{getCategoryDisplay(meaningfulCategories[0][0]).name}</strong> ({meaningfulCategories[0][1]} cases)</li>
          )}
          {meaningfulCategories.some(([cat]) => cat.includes('toxicity')) && (
            <li>• High toxicity detection - consider reviewing content guidelines</li>
          )}
          {meaningfulCategories.some(([cat]) => cat === 'Violence') && (
            <li>• Violence-related content detected - ensure safety policies are clear</li>
          )}
          {meaningfulCategories.some(([cat]) => cat.includes('indonesian')) && (
            <li>• Indonesian language content flagged - multilingual detection working</li>
          )}
          <li>• Monitor trends to identify emerging content patterns</li>
          <li>• Consider adjusting thresholds if blocking rates seem too high/low</li>
        </ul>
      </div>
    </div>
  );
};

// Cache Analysis Section Component
const CacheAnalysisSection = ({ apiClient }) => {
  const [cacheMetrics, setCacheMetrics] = useState(null);
  const [loading, setLoading] = useState(true);
  const [clearing, setClearing] = useState(false);
  
  useEffect(() => {
    if (!apiClient) {
      setLoading(false);
      return;
    }

    const fetchCacheMetrics = async () => {
      try {
        const response = await apiClient.get('/metrics');
        setCacheMetrics(response.data.cache);
      } catch (error) {
        console.error('Failed to fetch cache metrics:', error);
        if (error.response?.status === 403) {
          console.error('Authentication failed - API key may be invalid');
        }
      }
      setLoading(false);
    };
    
    fetchCacheMetrics();
    const interval = setInterval(fetchCacheMetrics, 10000); // Refresh every 10s
    return () => clearInterval(interval);
  }, [apiClient]);
  
  const refreshCacheMetrics = async () => {
    if (!apiClient) return;
    
    try {
      const response = await apiClient.get('/metrics');
      setCacheMetrics(response.data.cache);
    } catch (error) {
      console.error('Failed to fetch cache metrics:', error);
      if (error.response?.status === 403) {
        console.error('Authentication failed - API key may be invalid');
      }
    }
  };

  const clearCache = async () => {
    if (!apiClient) return;
    
    setClearing(true);
    try {
      await apiClient.delete('/cache');
      await refreshCacheMetrics(); // Refresh metrics after clearing
    } catch (error) {
      console.error('Failed to clear cache:', error);
      if (error.response?.status === 403) {
        console.error('Authentication failed - API key may be invalid');
      }
    }
    setClearing(false);
  };
  
  if (loading) {
    return (
      <div className="bg-white rounded-lg shadow p-6">
        <h3 className="text-lg font-semibold mb-4 flex items-center">
          <Database className="w-5 h-5 mr-2" />
          Cache Analysis
        </h3>
        <div className="text-center py-4">Loading cache metrics...</div>
      </div>
    );
  }
  
  const totalRequests = (cacheMetrics?.hits || 0) + (cacheMetrics?.misses || 0);
  const hitRate = cacheMetrics?.hit_rate_pct || 0;
  
  return (
    <div className="bg-white rounded-lg shadow p-6">
      <div className="flex justify-between items-center mb-4">
        <h3 className="text-lg font-semibold flex items-center">
          <Database className="w-5 h-5 mr-2" />
          Cache Analysis
        </h3>
        <div className="flex space-x-2">
          <button
            onClick={refreshCacheMetrics}
            disabled={loading}
            className="flex items-center px-3 py-1 text-sm bg-blue-100 text-blue-700 rounded-md hover:bg-blue-200 disabled:opacity-50"
          >
            <RefreshCw className={`w-4 h-4 mr-1 ${loading ? 'animate-spin' : ''}`} />
            Refresh
          </button>
          <button
            onClick={clearCache}
            disabled={clearing || totalRequests === 0}
            className="flex items-center px-3 py-1 text-sm bg-red-100 text-red-700 rounded-md hover:bg-red-200 disabled:opacity-50"
          >
            <Trash2 className="w-4 h-4 mr-1" />
            {clearing ? 'Clearing...' : 'Clear Cache'}
          </button>
        </div>
      </div>
      
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mb-6">
        <div className="bg-blue-50 p-4 rounded-lg">
          <div className="text-2xl font-bold text-blue-600">{cacheMetrics?.hits || 0}</div>
          <div className="text-sm text-blue-600">Cache Hits</div>
        </div>
        <div className="bg-orange-50 p-4 rounded-lg">
          <div className="text-2xl font-bold text-orange-600">{cacheMetrics?.misses || 0}</div>
          <div className="text-sm text-orange-600">Cache Misses</div>
        </div>
        <div className="bg-green-50 p-4 rounded-lg">
          <div className="text-2xl font-bold text-green-600">{totalRequests}</div>
          <div className="text-sm text-green-600">Total Requests</div>
        </div>
        <div className="bg-purple-50 p-4 rounded-lg">
          <div className="text-2xl font-bold text-purple-600">{hitRate.toFixed(1)}%</div>
          <div className="text-sm text-purple-600">Hit Rate</div>
        </div>
      </div>
      
      <div className="space-y-4">
        <div>
          <div className="flex justify-between items-center mb-2">
            <span className="text-sm font-medium">Cache Efficiency</span>
            <span className="text-sm text-gray-600">{hitRate.toFixed(1)}%</span>
          </div>
          <div className="w-full bg-gray-200 rounded-full h-2">
            <div 
              className={`h-2 rounded-full transition-all duration-300 ${
                hitRate >= 80 ? 'bg-green-500' : 
                hitRate >= 60 ? 'bg-yellow-500' : 
                'bg-red-500'
              }`}
              style={{ width: `${Math.min(hitRate, 100)}%` }}
            ></div>
          </div>
          <div className="flex justify-between text-xs text-gray-500 mt-1">
            <span>0%</span>
            <span className="text-center">
              {hitRate >= 80 ? 'Excellent' : 
               hitRate >= 60 ? 'Good' : 
               hitRate >= 40 ? 'Fair' : 'Poor'}
            </span>
            <span>100%</span>
          </div>
        </div>
        
        <div className="bg-gray-50 p-4 rounded-lg">
          <h4 className="font-medium mb-2">Cache Performance Insights</h4>
          <ul className="text-sm text-gray-600 space-y-1">
            {hitRate >= 80 && (
              <li className="flex items-center text-green-600">
                <CheckCircle className="w-4 h-4 mr-2" />
                Excellent cache performance - most requests are being served from cache
              </li>
            )}
            {hitRate < 80 && hitRate >= 60 && (
              <li className="flex items-center text-yellow-600">
                <AlertTriangle className="w-4 h-4 mr-2" />
                Good cache performance - consider optimizing cache strategy
              </li>
            )}
            {hitRate < 60 && totalRequests > 0 && (
              <li className="flex items-center text-red-600">
                <XCircle className="w-4 h-4 mr-2" />
                Low cache efficiency - review caching configuration
              </li>
            )}
            {totalRequests === 0 && (
              <li className="flex items-center text-gray-500">
                <Database className="w-4 h-4 mr-2" />
                No cache activity recorded yet
              </li>
            )}
            <li>• Cache helps reduce response time for duplicate prompts</li>
            <li>• Higher hit rates indicate better system efficiency</li>
            <li>• Clear cache if you notice stale results</li>
          </ul>
        </div>
      </div>
    </div>
  );
};

// Settings Tab
const SettingsTab = ({ apiClient }) => {
  const [config, setConfig] = useState(null);
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  
  useEffect(() => {
    if (!apiClient) {
      setLoading(false);
      return;
    }

    const fetchConfig = async () => {
      try {
        const response = await apiClient.get('/config');
        setConfig(response.data);
      } catch (error) {
        console.error('Failed to fetch config:', error);
        if (error.response?.status === 403) {
          console.error('Authentication failed - API key may be invalid');
        }
      }
      setLoading(false);
    };
    
    fetchConfig();
  }, [apiClient]);
  
  const updateConfig = async (updates) => {
    if (!apiClient) return;
    
    setSaving(true);
    try {
      await apiClient.put('/config', updates);
      // Refresh config
      const response = await apiClient.get('/config');
      setConfig(response.data);
    } catch (error) {
      console.error('Failed to update config:', error);
      if (error.response?.status === 403) {
        console.error('Authentication failed - API key may be invalid');
      }
    }
    setSaving(false);
  };
  
  const toggleMasterSwitch = (enabled) => {
    updateConfig({ enable_all: enabled });
  };
  
  const toggleGuard = (guardName, enabled) => {
    updateConfig({ guards: { [guardName]: enabled } });
  };
  
  const updateThreshold = (thresholdName, value) => {
    updateConfig({ thresholds: { [thresholdName]: value } });
  };
  
  if (loading) {
    return <div className="text-center py-8">Loading configuration...</div>;
  }
  
  if (!config) {
    return <div className="text-center py-8 text-red-600">Failed to load configuration</div>;
  }
  
  return (
    <div className="space-y-6">
      {/* Master Control */}
      <div className="bg-white rounded-lg shadow-md p-6 border border-gray-100">
        <h3 className="text-lg font-semibold mb-4 text-gray-800">Master Control</h3>
        <div className="flex items-center space-x-3 p-4 border rounded-lg hover:bg-gray-50">
          <input
            type="checkbox"
            id="master-control"
            checked={config.enable_all || false}
            onChange={(e) => !saving && toggleMasterSwitch(e.target.checked)}
            disabled={saving}
            className="w-5 h-5 text-blue-600 bg-gray-100 border-gray-300 rounded focus:ring-blue-500 focus:ring-2"
          />
          <label htmlFor="master-control" className="flex-1 cursor-pointer">
            <div className="font-medium text-gray-900">Enable All Guards</div>
            <div className="text-sm text-gray-600">Master switch to enable/disable all guards</div>
          </label>
          <div className={`px-3 py-1 rounded-full text-xs font-medium ${
            config.enable_all 
              ? 'bg-green-100 text-green-800' 
              : 'bg-gray-100 text-gray-600'
          }`}>
            {config.enable_all ? 'ENABLED' : 'DISABLED'}
          </div>
        </div>
      </div>
      
      {/* Individual Guard Controls */}
      <div className="bg-white rounded-lg shadow-md p-6 border border-gray-100">
        <h3 className="text-lg font-semibold mb-4 text-gray-800">Guard Selection</h3>
        <div className="space-y-3">
          {Object.entries(config.guards || {}).map(([guardName, enabled]) => {
            const isDisabled = saving || !config.enable_all;
            const guardId = `guard-${guardName}`;
            
            const getGuardDescription = (name) => {
              if (name.includes('llama_guard_8b')) return 'LLaMA Guard 8B - Advanced safety classifier';
              if (name.includes('llama_guard_1b')) return 'LLaMA Guard 1B - Lightweight safety classifier';
              if (name.includes('indobert')) return 'IndoBERT - Indonesian toxicity detection';
              if (name.includes('llm_guard')) return 'LLM Guard - Multi-validator prompt scanner';
              if (name.includes('nemo')) return 'NeMo Guardrails - Policy engine';
              return 'Content safety guard';
            };
            
            return (
              <div 
                key={guardName} 
                className={`flex items-center space-x-3 p-4 border rounded-lg transition-colors ${
                  isDisabled 
                    ? 'bg-gray-50 border-gray-200 opacity-60' 
                    : 'hover:bg-gray-50 border-gray-200'
                }`}
              >
                <input
                  type="checkbox"
                  id={guardId}
                  checked={enabled || false}
                  onChange={(e) => !isDisabled && toggleGuard(guardName, e.target.checked)}
                  disabled={isDisabled}
                  className="w-5 h-5 text-blue-600 bg-gray-100 border-gray-300 rounded focus:ring-blue-500 focus:ring-2 disabled:opacity-50"
                />
                <label htmlFor={guardId} className={`flex-1 ${
                  isDisabled ? 'cursor-not-allowed' : 'cursor-pointer'
                }`}>
                  <div className={`font-medium ${
                    isDisabled ? 'text-gray-400' : 'text-gray-900'
                  }`}>
                    {guardName.replace(/_/g, ' ').toUpperCase()}
                  </div>
                  <div className={`text-sm ${
                    isDisabled ? 'text-gray-400' : 'text-gray-600'
                  }`}>
                    {getGuardDescription(guardName)}
                  </div>
                  {isDisabled && !config.enable_all && (
                    <div className="text-xs text-amber-600 mt-1 font-medium">⚠️ Disabled by master switch</div>
                  )}
                </label>
                <div className={`px-3 py-1 rounded-full text-xs font-medium ${
                  isDisabled
                    ? 'bg-gray-100 text-gray-500'
                    : enabled 
                    ? 'bg-green-100 text-green-800' 
                    : 'bg-gray-100 text-gray-600'
                }`}>
                  {enabled ? 'ENABLED' : 'DISABLED'}
                </div>
              </div>
            );
          })}
        </div>
        
        {!config.enable_all && (
          <div className="mt-4 p-3 bg-amber-50 border border-amber-200 rounded-lg">
            <div className="flex items-center">
              <AlertTriangle className="w-4 h-4 text-amber-600 mr-2" />
              <span className="text-sm text-amber-700 font-medium">
                Individual guards are disabled. Enable the master switch to activate them.
              </span>
            </div>
          </div>
        )}
      </div>
      
      {/* Threshold Settings */}
      <div className="bg-white rounded-lg shadow p-6">
        <h3 className="text-lg font-semibold mb-4">Threshold Settings</h3>
        <div className="space-y-6">
          {Object.entries(config.thresholds || {}).map(([thresholdName, value]) => (
            <div key={thresholdName}>
              <div className="flex justify-between items-center mb-2">
                <label className="font-medium">
                  {thresholdName.replace('_', ' ').toUpperCase()}
                </label>
                <span className="text-sm text-gray-600">{value.toFixed(2)}</span>
              </div>
              <input
                type="range"
                min="0"
                max="1"
                step="0.01"
                value={value}
                onChange={(e) => updateThreshold(thresholdName, parseFloat(e.target.value))}
                className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer slider"
                disabled={saving}
              />
              <div className="flex justify-between text-xs text-gray-500 mt-1">
                <span>0.00</span>
                <span>1.00</span>
              </div>
            </div>
          ))}
        </div>
      </div>
      
      {/* Guard Versions */}
      {config.guard_versions && (
        <div className="bg-white rounded-lg shadow p-6">
          <h3 className="text-lg font-semibold mb-4">Guard Versions</h3>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {Object.entries(config.guard_versions).map(([guardName, version]) => (
              <div key={guardName} className="flex justify-between items-center py-2 border-b">
                <span className="font-medium">{guardName.replace('_', ' ').toUpperCase()}</span>
                <span className="text-sm text-gray-600">v{version}</span>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
};

// Main App Component
function App() {
  const [activeTab, setActiveTab] = useState('analysis');
  const [healthStatus, setHealthStatus] = useState({ status: 'checking' });
  
  // Health check effect
  useEffect(() => {
    const checkHealth = async () => {
      try {
        console.log('Starting health check...');
        setHealthStatus({ status: 'checking' });
        const response = await apiClient.get('/healthz');
        console.log('Health check response:', response.data);
        setHealthStatus(response.data);
      } catch (error) {
        console.error('Health check error:', error);
        setHealthStatus({ status: 'unhealthy', error: error.message });
      }
    };
    
    // Initial health check
    checkHealth();
    
    // Check every minute
    const interval = setInterval(checkHealth, 60000);
    
    return () => {
      clearInterval(interval);
    };
  }, []);
  
  const tabs = [
    { id: 'analysis', label: 'Input & Analysis', icon: Shield, component: AnalysisTab },
    { id: 'dashboard', label: 'Dashboard', icon: BarChart3, component: DashboardTab },
    { id: 'settings', label: 'Settings', icon: Settings, component: SettingsTab }
  ];
  
  const ActiveComponent = tabs.find(tab => tab.id === activeTab)?.component || AnalysisTab;
  

  
  return (
    <div className="min-h-screen bg-gray-50">
      {/* Header */}
      <header className="bg-white shadow">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center py-6">
            <div className="flex items-center">
              <Shield className="w-8 h-8 text-blue-600 mr-3" />
              <h1 className="text-2xl font-bold text-gray-900">Prompt Railguarding API</h1>
            </div>
            
            <div className="flex items-center space-x-4">
              {healthStatus && (
                <div className="flex items-center space-x-2">
                  <div className={`w-3 h-3 rounded-full ${
                    healthStatus.status === 'healthy' ? 'bg-green-500' :
                    healthStatus.status === 'degraded' ? 'bg-yellow-500' :
                    healthStatus.status === 'checking' ? 'bg-blue-500 animate-pulse' : 'bg-red-500'
                  }`}></div>
                  <span className="text-sm text-gray-600 capitalize">
                    {healthStatus.status === 'checking' ? 'Connecting...' : healthStatus.status}
                  </span>
                </div>
              )}
              

            </div>
          </div>
        </div>
      </header>
      
      {/* Navigation */}
      <nav className="bg-white border-b">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex space-x-4 py-4">
            {tabs.map(tab => (
              <TabButton
                key={tab.id}
                active={activeTab === tab.id}
                onClick={() => setActiveTab(tab.id)}
                icon={tab.icon}
              >
                {tab.label}
              </TabButton>
            ))}
          </div>
        </div>
      </nav>
      
      {/* Main Content */}
       <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
         <ActiveComponent apiClient={apiClient} />
       </main>
     </div>
  );
}

export default App;