import React, { useState, useEffect, useCallback } from 'react';
import { useNavigate } from 'react-router-dom';
import { ArrowLeft, Database, Search, CheckCircle, Clock, Eye } from 'lucide-react';
import { useTranslation } from '../contexts/TranslationContext';
import Card from './ui/Card';
import Button from './ui/Button';
import LoadingSpinner from './ui/LoadingSpinner';
import { TranslationService } from '../services/TranslationService';

function KnowledgeBasePage() {
  const navigate = useNavigate();
  const { t, currentLanguage } = useTranslation();
  const [items, setItems] = useState([]);
  const [stats, setStats] = useState({
    total_items: 0,
    total_workflows: 0,
    total_matches: 0,
    match_rate: 0
  });
  const [pendingItems, setPendingItems] = useState([]);
  const [loading, setLoading] = useState(true);
  const [searchQuery, setSearchQuery] = useState('');
  const [activeTab, setActiveTab] = useState('items');

  const loadKnowledgeBaseData = useCallback(async () => {
    try {
      setLoading(true);
      const params = new URLSearchParams();
      if (searchQuery) params.append('search', searchQuery);
      params.append('limit', '50');
      const response = await fetch(`/api/knowledge-base?${params.toString()}`);
      if (response.ok) {
        const data = await response.json();
        console.log('API response for knowledge base:', data); // Debugging log
        setItems(data.items || []);
        setStats(data.stats || stats);
      }
    } catch (error) {
      console.error('Failed to load knowledge base data:', error);
    } finally {
      setLoading(false);
    }
  }, [searchQuery]);

  const loadPendingApprovals = useCallback(async () => {
    try {
      const response = await fetch('/api/knowledge-base/pending');
      if (response.ok) {
        const data = await response.json();
        console.log('API response for pending approvals:', data); // Debugging log
        setPendingItems(data.pending_items || []);
      }
    } catch (error) {
      console.error('Failed to load pending approvals:', error);
    }
  }, []);

  useEffect(() => {
    loadKnowledgeBaseData();
    loadPendingApprovals();
    
    // Set up polling to refresh data every 10 seconds
    const intervalId = setInterval(() => {
      loadKnowledgeBaseData();
      loadPendingApprovals();
    }, 10000); // Poll every 10 seconds

    return () => clearInterval(intervalId);
  }, [loadKnowledgeBaseData, loadPendingApprovals]);

  const handleSearch = (e) => {
    setSearchQuery(e.target.value);
  };

  const formatDate = (dateString) => {
    try {
      return new Date(dateString).toLocaleDateString(
        currentLanguage === 'ja' ? 'ja-JP' : 'en-US'
      );
    } catch {
      return 'Invalid date';
    }
  };

  const formatNumber = (num) => {
    if (num == null || isNaN(num)) return '0';
    const locale = currentLanguage === 'ja' ? 'ja-JP' : 'en-US';
    return new Intl.NumberFormat(locale).format(num);
  };

  return (
    <div className="min-h-screen bg-gray-50">
      <div className="max-w-7xl mx-auto p-6">
        <div className="mb-8">
          <button onClick={() => navigate('/dashboard')} className="inline-flex items-center text-sm text-gray-500 hover:text-gray-700 mb-4">
            <ArrowLeft className="h-4 w-4 mr-1" />
            {t('common.back')} {t('common.to')} {t('navigation.dashboard')}
          </button>
          <h1 className="text-3xl font-bold text-gray-900">{t('knowledgeBase.title')}</h1>
          <p className="text-gray-600 mt-1">{t('knowledgeBase.subtitle')}</p>
        </div>
        <div className="grid grid-cols-1 md:grid-cols-4 gap-6 mb-8">
          <Card className="p-6">
            <div className="flex items-center">
              <Database className="h-8 w-8 text-blue-600" />
              <div className="ml-4">
                <p className="text-sm font-medium text-gray-500">{t('knowledgeBase.totalItems')}</p>
                <p className="text-2xl font-semibold text-gray-900">{formatNumber(stats.total_items)}</p>
              </div>
            </div>
          </Card>
          <Card className="p-6">
            <div className="flex items-center">
              <div className="h-8 w-8 bg-green-100 rounded-lg flex items-center justify-center">
                <span className="text-green-600 font-semibold">📋</span>
              </div>
              <div className="ml-4">
                <p className="text-sm font-medium text-gray-500">{t('knowledgeBase.totalWorkflows')}</p>
                <p className="text-2xl font-semibold text-gray-900">{formatNumber(stats.total_workflows)}</p>
              </div>
            </div>
          </Card>
          <Card className="p-6">
            <div className="flex items-center">
              <div className="h-8 w-8 bg-purple-100 rounded-lg flex items-center justify-center">
                <span className="text-purple-600 font-semibold">🔗</span>
              </div>
              <div className="ml-4">
                <p className="text-sm font-medium text-gray-500">{t('knowledgeBase.totalMatches')}</p>
                <p className="text-2xl font-semibold text-gray-900">{formatNumber(stats.total_matches)}</p>
              </div>
            </div>
          </Card>
          <Card className="p-6">
            <div className="flex items-center">
              <div className="h-8 w-8 bg-yellow-100 rounded-lg flex items-center justify-center">
                <span className="text-yellow-600 font-semibold">📊</span>
              </div>
              <div className="ml-4">
                <p className="text-sm font-medium text-gray-500">{t('knowledgeBase.matchRate')}</p>
                <p className="text-2xl font-semibold text-gray-900">{stats.match_rate}%</p>
              </div>
            </div>
          </Card>
        </div>
        <div className="border-b border-gray-200 mb-6">
          <nav className="-mb-px flex space-x-8">
            <button onClick={() => setActiveTab('items')} className={`py-2 px-1 border-b-2 font-medium text-sm ${
                activeTab === 'items'
                  ? 'border-primary-500 text-primary-600'
                  : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
              }`}>
              <Database className="h-4 w-4 inline mr-2" />
              {t('knowledgeBase.title')} ({formatNumber(stats.total_items)})
            </button>
            <button onClick={() => setActiveTab('pending')} className={`py-2 px-1 border-b-2 font-medium text-sm ${
                activeTab === 'pending'
                  ? 'border-primary-500 text-primary-600'
                  : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
              }`}>
              <Clock className="h-4 w-4 inline mr-2" />
              {t('knowledgeBase.pendingApprovals')} ({formatNumber(pendingItems.length)})
              {pendingItems.length > 0 && (
                <span className="ml-1 inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-yellow-100 text-yellow-800">
                  {pendingItems.length}
                </span>
              )}
            </button>
          </nav>
        </div>
        {activeTab === 'items' && (
          <>
            <div className="mb-6">
              <div className="relative">
                <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none">
                  <Search className="h-5 w-5 text-gray-400" />
                </div>
                <input
                  type="text"
                  value={searchQuery}
                  onChange={handleSearch}
                  placeholder={t('knowledgeBase.searchItems')}
                  className="block w-full pl-10 pr-3 py-2 border border-gray-300 rounded-md leading-5 bg-white placeholder-gray-500 focus:outline-none focus:placeholder-gray-400 focus:ring-1 focus:ring-primary-500 focus:border-primary-500 sm:text-sm"
                />
              </div>
            </div>
            <Card className="overflow-hidden">
              {loading ? (
                <div className="flex justify-center items-center py-12">
                  <LoadingSpinner size="lg" />
                </div>
              ) : items.length > 0 ? (
                <div className="overflow-x-auto">
                  <table className="min-w-full divide-y divide-gray-200">
                    <thead className="bg-gray-50">
                      <tr>
                        <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                          {t('results.columns.materialName')}
                        </th>
                        <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                          {t('results.columns.partNumber')}
                        </th>
                        <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                          {t('results.columns.classification')}
                        </th>
                        <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                          {t('results.columns.confidence')}
                        </th>
                        <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                          {t('common.created')}
                        </th>
                        <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                          {t('results.columns.supplierMatch')}
                        </th>
                      </tr>
                    </thead>
                    <tbody className="bg-white divide-y divide-gray-200">
                      {items.map((item) => (
                        <tr key={item.id} className="hover:bg-gray-50">
                          <td className="px-6 py-4 text-sm text-gray-900">
                            <div className="font-medium">{item.material_name}</div>
                            {item.description && (
                              <div className="text-xs text-gray-500 mt-1">{item.description}</div>
                            )}
                          </td>
                          <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                            {item.part_number || '-'}
                          </td>
                          <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                            {item.classification_label ? (
                              <div>
                                <span className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-blue-100 text-blue-800">
                                  {TranslationService.translateClassificationLabel(item.classification_label, currentLanguage)}
                                </span>
                              </div>
                            ) : (
                              '-'
                            )}
                          </td>
                          <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                            {item.confidence_level ? (
                              <span className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium ${
                                item.confidence_level === 'high'
                                  ? 'bg-green-100 text-green-800'
                                  : item.confidence_level === 'medium'
                                  ? 'bg-yellow-100 text-yellow-800'
                                  : 'bg-red-100 text-red-800'
                              }`}>
                                {TranslationService.translateConfidenceLevel(item.confidence_level, currentLanguage)}
                              </span>
                            ) : (
                              '-'
                            )}
                          </td>
                          <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                            {formatDate(item.created_at)}
                          </td>
                          <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                            {item.workflow_id ? (
                              <div>
                                <span className="text-blue-600">Workflow</span>
                                <div className="text-xs">{item.workflow_id.substring(0, 8)}...</div>
                              </div>
                            ) : (
                              'Manual'
                            )}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              ) : (
                <div className="text-center py-12">
                  <Database className="mx-auto h-12 w-12 text-gray-400" />
                  <h3 className="mt-2 text-sm font-medium text-gray-900">{t('knowledgeBase.noItems')}</h3>
                  <p className="mt-1 text-sm text-gray-500">{t('knowledgeBase.noItemsDescription')}</p>
                </div>
              )}
            </Card>
          </>
        )}
        {activeTab === 'pending' && (
          <Card className="overflow-hidden">
            {pendingItems.length > 0 ? (
              <div className="p-6">
                <div className="mb-4">
                  <h3 className="text-lg font-medium text-gray-900">{t('knowledgeBase.pendingApprovals')}</h3>
                  <p className="text-sm text-gray-600 mt-1">
                    These items are waiting for approval to be added to the knowledge base.
                    Go to the respective workflow results page to approve or reject them.
                  </p>
                </div>
                <div className="space-y-4">
                  {pendingItems.map((item) => {
                    const data = item.parsed_data || {};
                    return (
                      <div key={item.id} className="border border-gray-200 rounded-lg p-4 hover:bg-gray-50">
                        <div className="flex items-start justify-between">
                          <div className="flex-1">
                            <h4 className="font-medium text-gray-900">
                              {data.qa_material_name || 'Unknown Material'}
                            </h4>
                            <div className="mt-2 grid grid-cols-2 md:grid-cols-4 gap-4 text-sm text-gray-600">
                              <div>
                                <span className="font-medium">{t('results.columns.partNumber')}:</span>
                                <div>{data.part_number || 'N/A'}</div>
                              </div>
                              <div>
                                <span className="font-medium">{t('results.columns.confidence')}:</span>
                                <div>{Math.round((data.confidence_score || 0) * 100)}%</div>
                              </div>
                              <div>
                                <span className="font-medium">{t('results.columns.classification')}:</span>
                                <div>{TranslationService.translateClassificationLabel(data.qa_classification_label, currentLanguage)}</div>
                              </div>
                              <div>
                                <span className="font-medium">{t('results.columns.supplierMatch')}:</span>
                                <div>{TranslationService.translateMatchSource(data.match_source, currentLanguage)}</div>
                              </div>
                            </div>
                            {data.reasoning && (
                              <div className="mt-2 text-sm text-gray-600">
                                <span className="font-medium">{t('results.columns.reason')}:</span>
                                <p className="italic">"{data.reasoning}"</p>
                              </div>
                            )}
                            <div className="mt-2 text-xs text-gray-500">
                              Workflow: {item.workflow_id} • {t('common.created')}: {formatDate(item.created_at)}
                            </div>
                          </div>
                          <div className="ml-4">
                            <Button size="sm" variant="outline" onClick={() => navigate(`/results/${item.workflow_id}`)}>
                              <Eye className="h-4 w-4 mr-1" />
                              {t('knowledgeBase.review')}
                            </Button>
                          </div>
                        </div>
                      </div>
                    );
                  })}
                </div>
              </div>
            ) : (
              <div className="text-center py-12">
                <CheckCircle className="mx-auto h-12 w-12 text-green-400" />
                <h3 className="mt-2 text-sm font-medium text-gray-900">No Pending Approvals</h3>
                <p className="mt-1 text-sm text-gray-500">
                  All items have been reviewed. New items will appear here after workflow completion.
                </p>
              </div>
            )}
          </Card>
        )}
      </div>
    </div>
  );
}

export default KnowledgeBasePage;
