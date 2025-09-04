import React, { useState, useEffect, useCallback } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { ArrowLeft, Download, CheckCircle, X, Database, AlertCircle } from 'lucide-react';
import { useTranslation } from '../contexts/TranslationContext';
import { TranslationService } from '../services/TranslationService';
import Card from '../components/ui/Card';
import Button from '../components/ui/Button';
import toast from 'react-hot-toast';

// Define the base URL for the API calls.
const BASE_URL = 'http://localhost:8000';

function ResultsPage() {
  const { workflowId } = useParams();
  const navigate = useNavigate();
  const { t, currentLanguage } = useTranslation();
  const [results, setResults] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [pendingItems, setPendingItems] = useState([]);
  const [selectedItems, setSelectedItems] = useState(new Set());
  const [showApprovalSection, setShowApprovalSection] = useState(false);

  const loadResults = useCallback(async () => {
    try {
      setLoading(true);
      const response = await fetch(`/api/autonomous/workflow/${workflowId}/results`);
      if (!response.ok) throw new Error('Failed to fetch results');
      
      const data = await response.json();
      setResults(data);
    } catch (error) {
      console.error('Error fetching results:', error);
      setError(error.message);
    } finally {
      setLoading(false);
    }
  }, [workflowId]);

  const loadPendingApprovals = useCallback(async () => {
    try {
      const response = await fetch('/api/knowledge-base/pending');
      if (response.ok) {
        const data = await response.json();
        const workflowPendingItems = data.pending_items?.filter(
          item => item.workflow_id === workflowId
        ) || [];
        setPendingItems(workflowPendingItems);
        setShowApprovalSection(workflowPendingItems.length > 0);
      }
    } catch (error) {
      console.error('Error fetching pending approvals:', error);
    }
  }, [workflowId]);

  useEffect(() => {
    if (workflowId) {
      loadResults();
      loadPendingApprovals();
    }
  }, [workflowId, loadResults, loadPendingApprovals]);

  const handleItemSelection = (itemId) => {
    const newSelected = new Set(selectedItems);
    if (newSelected.has(itemId)) {
      newSelected.delete(itemId);
    } else {
      newSelected.add(itemId);
    }
    setSelectedItems(newSelected);
  };

  const handleApproveSelected = async () => {
    if (selectedItems.size === 0) {
      toast.error('Please select items to approve');
      return;
    }

    try {
      const response = await fetch(`${BASE_URL}/api/knowledge-base/approve?workflow_id=${workflowId}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          item_ids: Array.from(selectedItems).map(id => parseInt(id, 10))
        })
      });

      if (response.ok) {
        const data = await response.json();
        toast.success(`${data.approved_count} items approved for knowledge base`);
        loadPendingApprovals();
        setSelectedItems(new Set());
      } else {
        throw new Error('Failed to approve items');
      }
    } catch (error) {
      console.error('Error approving items:', error);
      toast.error('Failed to approve items');
    }
  };

  const handleRejectSelected = async () => {
    if (selectedItems.size === 0) {
      toast.error('Please select items to reject');
      return;
    }

    try {
      const response = await fetch(`${BASE_URL}/api/knowledge-base/reject?workflow_id=${workflowId}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          item_ids: Array.from(selectedItems).map(id => parseInt(id, 10))
        })
      });

      if (response.ok) {
        const data = await response.json();
        toast.success(`${data.rejected_count} items rejected`);
        loadPendingApprovals();
        setSelectedItems(new Set());
      } else {
        throw new Error('Failed to reject items');
      }
    } catch (error) {
      console.error('Error rejecting items:', error);
      toast.error('Failed to reject items');
    }
  };

  const exportResults = async () => {
    try {
      const exportData = {
        workflow_id: workflowId,
        results: results,
        export_date: new Date().toISOString(),
        language: currentLanguage
      };

      const blob = new Blob([JSON.stringify(exportData, null, 2)], {
        type: 'application/json'
      });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `bom-results-${workflowId}-${currentLanguage}.json`;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);

      toast.success(t('results.resultsExported'));
    } catch (error) {
      console.error('Export failed:', error);
      toast.error('Export failed');
    }
  };

  if (loading) {
    return (
      <div className="min-h-screen bg-gray-50 flex items-center justify-center">
        <div className="text-center">
          <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary-600 mx-auto mb-4"></div>
          <p className="text-gray-600">{t('common.loading')}</p>
        </div>
      </div>
    );
  }

  if (error || !results) {
    return (
      <div className="min-h-screen bg-gray-50 p-6">
        <div className="max-w-7xl mx-auto">
          <div className="mb-8">
            <button
              onClick={() => navigate('/dashboard')}
              className="inline-flex items-center text-sm text-gray-500 hover:text-gray-700"
            >
              <ArrowLeft className="h-4 w-4 mr-1" />
              Back to Dashboard
            </button>
          </div>
          <Card className="p-6 text-center">
            <div className="text-red-600 mb-4">
              <div className="w-16 h-16 mx-auto mb-4 bg-red-100 rounded-full flex items-center justify-center">
                <AlertCircle className="h-8 w-8" />
              </div>
            </div>
            <h3 className="text-lg font-medium text-gray-900 mb-2">Error</h3>
            <p className="text-gray-600">Failed to load results</p>
          </Card>
        </div>
      </div>
    );
  }

  const matches = results.matches || [];
  const summary = results.summary || {};

  return (
    <div className="min-h-screen bg-gray-50">
      <div className="max-w-7xl mx-auto p-6">
        <div className="mb-8">
          <div className="flex items-center justify-between">
            <div>
              <button
                onClick={() => navigate('/dashboard')}
                className="inline-flex items-center text-sm text-gray-500 hover:text-gray-700 mb-4"
              >
                <ArrowLeft className="h-4 w-4 mr-1" />
                Back to Dashboard
              </button>
              <h1 className="text-3xl font-bold text-gray-900">{t('results.title')}</h1>
              <p className="text-gray-600 mt-1">
                {t('results.workflowId')}: {workflowId} • {t('results.withItemClassificationReasons')}
              </p>
            </div>
            <div className="flex items-center space-x-3">
              <Button onClick={exportResults} variant="outline">
                <Download className="h-4 w-4 mr-2" />
                {t('results.exportResults')}
              </Button>
            </div>
          </div>
        </div>
        {showApprovalSection && (
          <Card className="p-6 mb-8 border-amber-200 bg-amber-50">
            <div className="flex items-start space-x-3">
              <Database className="h-6 w-6 text-amber-600 mt-0.5" />
              <div className="flex-1">
                <h3 className="text-lg font-medium text-gray-900 mb-2">
                  {t('knowledgeBase.pendingApprovals')}
                </h3>
                <p className="text-sm text-gray-600 mb-4">
                  {pendingItems.length} items from this workflow are pending approval to be added to the knowledge base. 
                  Please review and approve items that should be saved for future comparisons.
                </p>
                
                <div className="space-y-3 mb-4">
                  {pendingItems.map((item) => {
                    const data = item.parsed_data || {};
                    return (
                      <div
                        key={item.id}
                        className={`p-3 rounded-lg border cursor-pointer transition-colors ${
                          selectedItems.has(item.id)
                            ? 'bg-primary-50 border-primary-300'
                            : 'bg-white border-gray-200 hover:border-gray-300'
                        }`}
                        onClick={() => handleItemSelection(item.id)}
                      >
                        <div className="flex items-start justify-between">
                          <div className="flex-1">
                            <div className="flex items-center space-x-2">
                              <input
                                type="checkbox"
                                checked={selectedItems.has(item.id)}
                                onChange={() => handleItemSelection(item.id)}
                                className="rounded border-gray-300 text-primary-600 focus:ring-primary-500"
                              />
                              <h4 className="font-medium text-gray-900">
                                {data.qa_material_name || 'Unknown Material'}
                              </h4>
                            </div>
                            <div className="mt-1 ml-6 text-sm text-gray-600">
                              <span className="mr-4">{t('results.columns.partNumber')}: {data.part_number || 'N/A'}</span>
                              <span className="mr-4">
                                {t('results.columns.confidence')}: {Math.round((data.confidence_score || 0) * 100)}%
                              </span>
                              <span>
                                {t('results.columns.classification')}: {TranslationService.translateClassificationLabel(data.qa_classification_label, currentLanguage)}
                              </span>
                            </div>
                            {data.reasoning && (
                              <p className="mt-1 ml-6 text-sm text-gray-500 italic">
                                "{data.reasoning}"
                              </p>
                            )}
                          </div>
                        </div>
                      </div>
                    );
                  })}
                </div>

                <div className="flex items-center space-x-3">
                  <Button
                    onClick={handleApproveSelected}
                    disabled={selectedItems.size === 0}
                    className="bg-green-600 hover:bg-green-700 text-white"
                  >
                    <CheckCircle className="h-4 w-4 mr-2" />
                    Approve Selected ({selectedItems.size})
                  </Button>
                  <Button
                    onClick={handleRejectSelected}
                    disabled={selectedItems.size === 0}
                    variant="outline"
                    className="border-red-300 text-red-700 hover:bg-red-50"
                  >
                    <X className="h-4 w-4 mr-2" />
                    Reject Selected ({selectedItems.size})
                  </Button>
                  <Button
                    onClick={() => setSelectedItems(new Set(pendingItems.map(item => item.id)))}
                    variant="outline"
                    size="sm"
                  >
                    Select All
                  </Button>
                  <Button
                    onClick={() => setSelectedItems(new Set())}
                    variant="outline"
                    size="sm"
                  >
                    Clear Selection
                  </Button>
                </div>
              </div>
            </div>
          </Card>
        )}
        <div className="grid grid-cols-1 md:grid-cols-4 gap-6 mb-8">
          <Card className="p-6">
            <div className="flex items-center">
              <div className="flex-shrink-0">
                <div className="w-8 h-8 bg-blue-100 rounded-lg flex items-center justify-center">
                  <span className="text-blue-600 font-semibold text-sm">📊</span>
                </div>
              </div>
              <div className="ml-4">
                <p className="text-sm font-medium text-gray-500">{t('results.materialsProcessed')}</p>
                <p className="text-2xl font-semibold text-gray-900">
                  {summary.total_materials || 0}
                </p>
              </div>
            </div>
          </Card>
          <Card className="p-6">
            <div className="flex items-center">
              <div className="flex-shrink-0">
                <div className="w-8 h-8 bg-green-100 rounded-lg flex items-center justify-center">
                  <CheckCircle className="h-4 w-4 text-green-600" />
                </div>
              </div>
              <div className="ml-4">
                <p className="text-sm font-medium text-gray-500">{t('results.successfulMatches')}</p>
                <p className="text-2xl font-semibold text-gray-900">
                  {summary.successful_matches || 0}
                </p>
              </div>
            </div>
          </Card>
          <Card className="p-6">
            <div className="flex items-center">
              <div className="flex-shrink-0">
                <div className="w-8 h-8 bg-yellow-100 rounded-lg flex items-center justify-center">
                  <span className="text-yellow-600 font-semibold">📈</span>
                </div>
              </div>
              <div className="ml-4">
                <p className="text-sm font-medium text-gray-500">{t('results.averageConfidence')}</p>
                <p className="text-2xl font-semibold text-gray-900">
                  {matches.length > 0 ? Math.round((matches.reduce((sum, m) => sum + (m.confidence_score || 0), 0) / matches.length) * 100) : 0}%
                </p>
              </div>
            </div>
          </Card>
          <Card className="p-6">
            <div className="flex items-center">
              <div className="flex-shrink-0">
                <Database className="h-4 w-4 text-purple-600" />
              </div>
              <div className="ml-4">
                <p className="text-sm font-medium text-gray-500">{t('results.knowledgeBaseMatches')}</p>
                <p className="text-2xl font-semibold text-gray-900">
                  {summary.knowledge_base_matches || 0}
                </p>
              </div>
            </div>
          </Card>
        </div>
        <Card className="overflow-hidden">
          <div className="overflow-x-auto">
            <table className="min-w-full divide-y divide-gray-200">
              <thead className="bg-gray-50">
                <tr>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    {t('results.columns.sno')}
                  </th>
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
                    {t('results.columns.supplierMatch')}
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    {t('results.columns.reason')}
                  </th>
                </tr>
              </thead>
              <tbody className="bg-white divide-y divide-gray-200">
                {matches.length > 0 ? (
                  matches.map((match, index) => (
                    <tr key={index} className="hover:bg-gray-50">
                      <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                        {index + 1}
                      </td>
                      <td className="px-6 py-4 text-sm text-gray-900">
                        <div className="font-medium">{match.material_name}</div>
                        {match.qa_excerpt && (
                          <div className="text-xs text-gray-500 mt-1">
                            "{match.qa_excerpt}"
                          </div>
                        )}
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                        {match.part_number || '-'}
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                        <div className="flex items-center">
                          <span className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-gray-100 text-gray-800">
                            {TranslationService.translateClassificationLabel(match.qa_classification_label, currentLanguage)}
                          </span>
                        </div>
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                        <span className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium ${
                          match.qa_confidence_level === 'high' ? 'bg-green-100 text-green-800' :
                          match.qa_confidence_level === 'medium' ? 'bg-yellow-100 text-yellow-800' :
                          'bg-red-100 text-red-800'
                        }`}>
                          {TranslationService.translateConfidenceLevel(match.qa_confidence_level, currentLanguage)}
                        </span>
                      </td>
                      <td className="px-6 py-4 text-sm text-gray-900">
                        {match.confidence_score > 0 ? (
                          <div>
                            <div className="font-medium">{match.supplier_description}</div>
                            <div className="text-xs text-gray-500">
                              {match.supplier_part_number} • {Math.round(match.confidence_score * 100)}%
                            </div>
                            <div className="text-xs text-blue-600 mt-1">
                              {TranslationService.translateMatchSource(match.match_source, currentLanguage)}
                            </div>
                          </div>
                        ) : (
                          <span className="text-gray-400">-</span>
                        )}
                      </td>
                      <td className="px-6 py-4 text-sm text-gray-600">
                        {match.reasoning || t('results.noReasonProvided')}
                      </td>
                    </tr>
                  ))
                ) : (
                  <tr>
                    <td colSpan="7" className="px-6 py-8 text-center text-gray-500">
                      {t('results.noMaterialsMatch')}
                    </td>
                  </tr>
                )}
              </tbody>
            </table>
          </div>
        </Card>
      </div>
    </div>
  );
}

export default ResultsPage;
