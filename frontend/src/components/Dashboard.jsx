import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { Play, CheckCircle, Database, TrendingUp, Upload } from 'lucide-react';
import { useTranslation } from '../contexts/TranslationContext';
import Card from '../components/ui/Card';
import Button from '../components/ui/Button';
import LoadingSpinner from '../components/ui/LoadingSpinner';
import toast from 'react-hot-toast';

function Dashboard() {
  const navigate = useNavigate();
  const { t } = useTranslation();
  const [workflows, setWorkflows] = useState([]);
  const [stats, setStats] = useState({
    total: 0,
    completed: 0,
    processing: 0,
    success_rate: 0
  });
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    loadDashboardData();
    // Set up polling for real-time updates
    const interval = setInterval(loadDashboardData, 5000);
    return () => clearInterval(interval);
  }, []);

  const loadDashboardData = async () => {
    try {
      setLoading(true);
      const workflowsResponse = await fetch('/api/workflows');
      if (workflowsResponse.ok) {
        const workflowsData = await workflowsResponse.json();
        const workflowList = workflowsData.workflows || [];
        setWorkflows(workflowList);
        const totalWorkflows = workflowList.length;
        const completedWorkflows = workflowList.filter(w => w.status === 'completed').length;
        const processingWorkflows = workflowList.filter(w => w.status === 'processing').length;
        const successRate = totalWorkflows > 0 ? Math.round((completedWorkflows / totalWorkflows) * 100) : 0;
        setStats({
          total: totalWorkflows,
          completed: completedWorkflows,
          processing: processingWorkflows,
          success_rate: successRate
        });
      }
    } catch (error) {
      console.error('Failed to load dashboard data:', error);
      toast.error('Failed to load dashboard data');
    } finally {
      setLoading(false);
    }
  };

  const formatDate = (dateString) => {
    try {
      return new Date(dateString).toLocaleDateString();
    } catch {
      return 'Invalid date';
    }
  };

  const getStatusColor = (status) => {
    switch (status) {
      case 'completed': return 'bg-green-100 text-green-800';
      case 'processing': return 'bg-yellow-100 text-yellow-800';
      case 'error': return 'bg-red-100 text-red-800';
      default: return 'bg-gray-100 text-gray-800';
    }
  };

  if (loading && workflows.length === 0) {
    return (
      <div className="min-h-screen bg-gray-50 flex items-center justify-center">
        <div className="text-center">
          <LoadingSpinner size="lg" className="mx-auto mb-4" />
          <p className="text-gray-600">{t('common.loading')}</p>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gray-50">
      <div className="max-w-7xl mx-auto p-6">
        <div className="mb-8">
          <h1 className="text-3xl font-bold text-gray-900">{t('dashboard.title')}</h1>
          <p className="text-gray-600 mt-1">{t('dashboard.subtitle')}</p>
        </div>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
          <Card className="p-6">
            <div className="flex items-center">
              <div className="flex-shrink-0">
                <div className="w-8 h-8 bg-blue-100 rounded-lg flex items-center justify-center">
                  <Database className="h-5 w-5 text-blue-600" />
                </div>
              </div>
              <div className="ml-4">
                <p className="text-sm font-medium text-gray-500">Total Workflows</p>
                <p className="text-2xl font-semibold text-gray-900">{stats.total}</p>
              </div>
            </div>
          </Card>
          <Card className="p-6">
            <div className="flex items-center">
              <div className="flex-shrink-0">
                <div className="w-8 h-8 bg-green-100 rounded-lg flex items-center justify-center">
                  <CheckCircle className="h-5 w-5 text-green-600" />
                </div>
              </div>
              <div className="ml-4">
                <p className="text-sm font-medium text-gray-500">Completed</p>
                <p className="text-2xl font-semibold text-gray-900">{stats.completed}</p>
              </div>
            </div>
          </Card>
          <Card className="p-6">
            <div className="flex items-center">
              <div className="flex-shrink-0">
                <div className="w-8 h-8 bg-yellow-100 rounded-lg flex items-center justify-center">
                  <Play className="h-5 w-5 text-yellow-600" />
                </div>
              </div>
              <div className="ml-4">
                <p className="text-sm font-medium text-gray-500">Processing</p>
                <p className="text-2xl font-semibold text-gray-900">{stats.processing}</p>
              </div>
            </div>
          </Card>
          <Card className="p-6">
            <div className="flex items-center">
              <div className="flex-shrink-0">
                <div className="w-8 h-8 bg-purple-100 rounded-lg flex items-center justify-center">
                  <TrendingUp className="h-5 w-5 text-purple-600" />
                </div>
              </div>
              <div className="ml-4">
                <p className="text-sm font-medium text-gray-500">Success Rate</p>
                <p className="text-2xl font-semibold text-gray-900">{stats.success_rate}%</p>
              </div>
            </div>
          </Card>
        </div>
        <div className="mb-8">
          <Button onClick={() => navigate('/upload')} className="btn-primary">
            <Upload className="h-5 w-5 mr-2" />
            {t('dashboard.startProcessing')}
          </Button>
        </div>
        <Card>
          <div className="px-6 py-4 border-b border-gray-200">
            <h3 className="text-lg font-medium text-gray-900">Recent Workflows</h3>
          </div>
          <div className="p-6">
            {workflows.length > 0 ? (
              <div className="space-y-4">
                {workflows.slice(0, 10).map((workflow) => (
                  <div
                    key={workflow.id}
                    className="flex items-center justify-between p-4 border border-gray-200 rounded-lg hover:bg-gray-50"
                  >
                    <div className="flex-1">
                      <div className="flex items-center justify-between">
                        <h4 className="font-medium text-gray-900">
                          {workflow.id.substring(0, 8)}...
                        </h4>
                        <span className="text-sm text-gray-500">
                          {formatDate(workflow.created_at)}
                        </span>
                      </div>
                      <div className="mt-1 flex items-center space-x-4">
                        <span
                          className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium ${getStatusColor(
                            workflow.status
                          )}`}
                        >
                          {workflow.status}
                        </span>
                        <span className="text-sm text-gray-500">
                          Mode: {workflow.comparison_mode || 'full'}
                        </span>
                        {workflow.progress && (
                          <span className="text-sm text-gray-500">
                            {workflow.progress}%
                          </span>
                        )}
                      </div>
                      {workflow.message && (
                        <p className="text-sm text-gray-600 mt-1">{workflow.message}</p>
                      )}
                    </div>
                    <div className="flex items-center space-x-2 ml-4">
                      {workflow.status === 'completed' && workflow.has_results && (
                        <Button
                          size="sm"
                          variant="outline"
                          onClick={() => navigate(`/results/${workflow.id}`)}
                        >
                          View Results
                        </Button>
                      )}
                      {workflow.status === 'processing' && (
                        <Button
                          size="sm"
                          variant="outline"
                          onClick={() => navigate(`/processing/${workflow.id}`)}
                        >
                          View Status
                        </Button>
                      )}
                    </div>
                  </div>
                ))}
              </div>
            ) : (
              <div className="text-center py-8">
                <Database className="mx-auto h-12 w-12 text-gray-400" />
                <h3 className="mt-2 text-sm font-medium text-gray-900">No workflows yet</h3>
                <p className="mt-1 text-sm text-gray-500">
                  Start by uploading your first document.
                </p>
                <div className="mt-6">
                  <Button onClick={() => navigate('/upload')}>
                    <Upload className="h-4 w-4 mr-2" />
                    Start Processing
                  </Button>
                </div>
              </div>
            )}
          </div>
        </Card>
      </div>
    </div>
  );
}

export default Dashboard;
