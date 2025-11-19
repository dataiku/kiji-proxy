import React, { useState } from 'react';
import { Shield, Eye, EyeOff, ArrowRight, Send, AlertCircle, Copy, Check } from 'lucide-react';

export default function PrivacyProxyUI() {
  const [inputData, setInputData] = useState('');
  const [maskedInput, setMaskedInput] = useState('');
  const [maskedOutput, setMaskedOutput] = useState('');
  const [finalOutput, setFinalOutput] = useState('');
  const [isProcessing, setIsProcessing] = useState(false);
  const [detectedEntities, setDetectedEntities] = useState([]);
  const [activeView, setActiveView] = useState('flow');
  const [copiedStage, setCopiedStage] = useState(null);
  const [showConfidence, setShowConfidence] = useState(true);

  // Call the real /details endpoint
  const handleSubmit = async () => {
    if (!inputData.trim()) return;

    setIsProcessing(true);

    try {
      // Create OpenAI chat completion request format
      const requestBody = {
        model: "gpt-3.5-turbo",
        messages: [
          {
            role: "user",
            content: inputData
          }
        ],
        max_tokens: 1000
      };

      // Call the /details endpoint (proxied to Go server)
      const response = await fetch('/details', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(requestBody)
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data = await response.json();

      // Update state with real data from the API
      setMaskedInput(data.masked_request);
      setMaskedOutput(data.masked_response);
      setFinalOutput(data.unmasked_response);

      // Transform PII entities to match UI format
      const transformedEntities = data.pii_entities.map(entity => ({
        type: entity.label.toLowerCase(),
        original: entity.text,
        token: entity.masked_text,
        confidence: entity.confidence
      }));
      setDetectedEntities(transformedEntities);

    } catch (error) {
      console.error('Error calling /details endpoint:', error);
      alert(`Error: ${error.message}`);
    } finally {
      setIsProcessing(false);
    }
  };


  const handleReset = () => {
    setInputData('');
    setMaskedInput('');
    setMaskedOutput('');
    setFinalOutput('');
    setDetectedEntities([]);
    setCopiedStage(null);
  };

  const copyToClipboard = (text, stage) => {
    navigator.clipboard.writeText(text);
    setCopiedStage(stage);
    setTimeout(() => setCopiedStage(null), 2000);
  };

  const getConfidenceColor = (confidence) => {
    if (confidence >= 0.95) return 'text-green-700 bg-green-100';
    if (confidence >= 0.85) return 'text-blue-700 bg-blue-100';
    if (confidence >= 0.75) return 'text-yellow-700 bg-yellow-100';
    return 'text-orange-700 bg-orange-100';
  };

  const getConfidenceLabel = (confidence) => {
    if (confidence >= 0.95) return 'Very High';
    if (confidence >= 0.85) return 'High';
    if (confidence >= 0.75) return 'Medium';
    return 'Low';
  };

  const highlightDifferences = (original, modified, entities) => {
    let result = modified;
    entities.forEach(entity => {
      // Highlight the masked text in the modified version
      result = result.replace(
        entity.token,
        `<mark class="bg-yellow-200 px-1 rounded">${entity.token}</mark>`
      );
    });
    return result;
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 to-slate-100 p-4 md:p-8">
      <div className="max-w-7xl mx-auto">
        {/* Header */}
        <div className="text-center mb-8">
          <div className="flex items-center justify-center gap-3 mb-4">
            <Shield className="w-10 h-10 text-blue-600" />
            <h1 className="text-4xl font-bold text-slate-800">Yaak - Privacy Proxy</h1>
          </div>
          <p className="text-slate-600 text-lg">
            Secure data masking, AI processing, and demasking pipeline
          </p>
        </div>

        {/* View Toggle */}
        <div className="flex justify-center mb-6">
          <div className="bg-white rounded-lg shadow-sm p-1 inline-flex">
            <button
              onClick={() => setActiveView('flow')}
              className={`px-6 py-2 rounded-md transition-colors ${
                activeView === 'flow'
                  ? 'bg-blue-600 text-white'
                  : 'text-slate-600 hover:text-slate-800'
              }`}
            >
              Flow View
            </button>
            <button
              onClick={() => setActiveView('comparison')}
              className={`px-6 py-2 rounded-md transition-colors ${
                activeView === 'comparison'
                  ? 'bg-blue-600 text-white'
                  : 'text-slate-600 hover:text-slate-800'
              }`}
            >
              Side-by-Side
            </button>
            <button
              onClick={() => setActiveView('diff')}
              className={`px-6 py-2 rounded-md transition-colors ${
                activeView === 'diff'
                  ? 'bg-blue-600 text-white'
                  : 'text-slate-600 hover:text-slate-800'
              }`}
            >
              Diff View
            </button>
          </div>
        </div>

        {/* Input Section */}
        <div className="bg-white rounded-xl shadow-lg p-6 mb-6">
          <div className="flex items-center justify-between mb-4">
            <h2 className="text-xl font-semibold text-slate-800 flex items-center gap-2">
              <Eye className="w-5 h-5" />
              Input Data (A)
            </h2>
            <span className="text-sm text-slate-500">Original data with PII</span>
          </div>
          <textarea
            value={inputData}
            onChange={(e) => setInputData(e.target.value)}
            placeholder="Enter your message with sensitive information...&#10;&#10;Example: Hi, my name is John Smith and my email is john.smith@email.com. My phone is 555-123-4567.&#10;&#10;This will be processed through the real PII detection and masking pipeline."
            className="w-full h-32 p-4 border-2 border-slate-200 rounded-lg focus:border-blue-500 focus:outline-none resize-none font-mono text-sm"
          />
          <div className="flex gap-3 mt-4">
            <button
              onClick={handleSubmit}
              disabled={!inputData.trim() || isProcessing}
              className="flex items-center gap-2 px-6 py-3 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors font-medium"
            >
              {isProcessing ? (
                <>
                  <div className="w-5 h-5 border-2 border-white border-t-transparent rounded-full animate-spin" />
                  Processing...
                </>
              ) : (
                <>
                  <Send className="w-5 h-5" />
                  Process Data
                </>
              )}
            </button>
            <button
              onClick={handleReset}
              className="px-6 py-3 border-2 border-slate-300 text-slate-700 rounded-lg hover:bg-slate-50 transition-colors font-medium"
            >
              Reset
            </button>
          </div>
        </div>

        {/* Detected Entities */}
        {detectedEntities.length > 0 && (
          <div className="bg-amber-50 border-2 border-amber-200 rounded-xl p-6 mb-6">
            <div className="flex items-center justify-between mb-3">
              <div className="flex items-center gap-2">
                <AlertCircle className="w-5 h-5 text-amber-600" />
                <h3 className="text-lg font-semibold text-amber-900">
                  Detected & Masked Entities ({detectedEntities.length})
                </h3>
              </div>
              <button
                onClick={() => setShowConfidence(!showConfidence)}
                className="text-sm px-3 py-1 bg-white border border-amber-300 rounded-lg hover:bg-amber-50 transition-colors"
              >
                {showConfidence ? 'Hide' : 'Show'} Confidence
              </button>
            </div>
            <div className="flex flex-wrap gap-2">
              {detectedEntities.map((entity, idx) => (
                <div
                  key={idx}
                  className="inline-flex items-center gap-2 px-3 py-2 bg-white border border-amber-300 rounded-lg text-sm"
                >
                  <span className="font-semibold text-amber-700 uppercase text-xs">{entity.type}</span>
                  <span className="text-slate-700">{entity.original}</span>
                  <ArrowRight className="w-3 h-3 text-amber-400" />
                  <span className="font-mono text-amber-600">{entity.token}</span>
                  {showConfidence && (
                    <>
                      <span className="text-slate-300">|</span>
                      <span className={`text-xs font-semibold px-2 py-0.5 rounded ${getConfidenceColor(entity.confidence)}`}>
                        {getConfidenceLabel(entity.confidence)} ({(entity.confidence * 100).toFixed(1)}%)
                      </span>
                    </>
                  )}
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Flow View */}
        {activeView === 'flow' && maskedInput && (
          <div className="space-y-6">
            {/* Stage 1: Masked Input */}
            <div className="bg-white rounded-xl shadow-lg p-6 border-l-4 border-green-500">
              <div className="flex items-center justify-between mb-4">
                <h2 className="text-xl font-semibold text-slate-800 flex items-center gap-2">
                  <EyeOff className="w-5 h-5 text-green-600" />
                  Masked Input (A')
                </h2>
                <div className="flex items-center gap-2">
                  <span className="text-sm px-3 py-1 bg-green-100 text-green-700 rounded-full font-medium">
                    PII Protected
                  </span>
                  <button
                    onClick={() => copyToClipboard(maskedInput, 'A')}
                    className="p-2 hover:bg-slate-100 rounded-lg transition-colors"
                    title="Copy to clipboard"
                  >
                    {copiedStage === 'A' ? (
                      <Check className="w-4 h-4 text-green-600" />
                    ) : (
                      <Copy className="w-4 h-4 text-slate-500" />
                    )}
                  </button>
                </div>
              </div>
              <div className="bg-slate-50 rounded-lg p-4 font-mono text-sm text-slate-700 whitespace-pre-wrap border border-slate-200">
                {maskedInput}
              </div>
            </div>

            {/* Arrow */}
            <div className="flex justify-center">
              <div className="flex items-center gap-2 text-slate-400">
                <ArrowRight className="w-6 h-6" />
                <span className="text-sm font-medium">Sent to OpenAI API</span>
                <ArrowRight className="w-6 h-6" />
              </div>
            </div>

            {/* Stage 2: Masked Output */}
            <div className="bg-white rounded-xl shadow-lg p-6 border-l-4 border-purple-500">
              <div className="flex items-center justify-between mb-4">
                <h2 className="text-xl font-semibold text-slate-800 flex items-center gap-2">
                  <Shield className="w-5 h-5 text-purple-600" />
                  Masked Output (B')
                </h2>
                <div className="flex items-center gap-2">
                  <span className="text-sm px-3 py-1 bg-purple-100 text-purple-700 rounded-full font-medium">
                    From OpenAI
                  </span>
                  <button
                    onClick={() => copyToClipboard(maskedOutput, 'B')}
                    className="p-2 hover:bg-slate-100 rounded-lg transition-colors"
                    title="Copy to clipboard"
                  >
                    {copiedStage === 'B' ? (
                      <Check className="w-4 h-4 text-green-600" />
                    ) : (
                      <Copy className="w-4 h-4 text-slate-500" />
                    )}
                  </button>
                </div>
              </div>
              <div className="bg-slate-50 rounded-lg p-4 font-mono text-sm text-slate-700 whitespace-pre-wrap border border-slate-200">
                {maskedOutput}
              </div>
            </div>

            {/* Arrow */}
            <div className="flex justify-center">
              <div className="flex items-center gap-2 text-slate-400">
                <ArrowRight className="w-6 h-6" />
                <span className="text-sm font-medium">Demasked by Proxy</span>
                <ArrowRight className="w-6 h-6" />
              </div>
            </div>

            {/* Stage 3: Final Output */}
            <div className="bg-white rounded-xl shadow-lg p-6 border-l-4 border-blue-500">
              <div className="flex items-center justify-between mb-4">
                <h2 className="text-xl font-semibold text-slate-800 flex items-center gap-2">
                  <Eye className="w-5 h-5 text-blue-600" />
                  Final Output (B)
                </h2>
                <div className="flex items-center gap-2">
                  <span className="text-sm px-3 py-1 bg-blue-100 text-blue-700 rounded-full font-medium">
                    Restored Data
                  </span>
                  <button
                    onClick={() => copyToClipboard(finalOutput, 'final')}
                    className="p-2 hover:bg-slate-100 rounded-lg transition-colors"
                    title="Copy to clipboard"
                  >
                    {copiedStage === 'final' ? (
                      <Check className="w-4 h-4 text-green-600" />
                    ) : (
                      <Copy className="w-4 h-4 text-slate-500" />
                    )}
                  </button>
                </div>
              </div>
              <div className="bg-slate-50 rounded-lg p-4 font-mono text-sm text-slate-700 whitespace-pre-wrap border border-slate-200">
                {finalOutput}
              </div>
            </div>
          </div>
        )}

        {/* Comparison View */}
        {activeView === 'comparison' && maskedInput && (
          <div className="grid md:grid-cols-2 gap-6">
            {/* Left Column - Input */}
            <div className="space-y-6">
              <div className="bg-white rounded-xl shadow-lg p-6">
                <div className="flex items-center justify-between mb-4">
                  <h2 className="text-lg font-semibold text-slate-800 flex items-center gap-2">
                    <Eye className="w-5 h-5" />
                    Original (A)
                  </h2>
                  <button
                    onClick={() => copyToClipboard(inputData, 'orig')}
                    className="p-2 hover:bg-slate-100 rounded-lg transition-colors"
                  >
                    {copiedStage === 'orig' ? (
                      <Check className="w-4 h-4 text-green-600" />
                    ) : (
                      <Copy className="w-4 h-4 text-slate-500" />
                    )}
                  </button>
                </div>
                <div className="bg-red-50 rounded-lg p-4 font-mono text-sm text-slate-700 whitespace-pre-wrap border-2 border-red-200 min-h-32">
                  {inputData}
                </div>
              </div>

              <div className="bg-white rounded-xl shadow-lg p-6">
                <div className="flex items-center justify-between mb-4">
                  <h2 className="text-lg font-semibold text-slate-800 flex items-center gap-2">
                    <EyeOff className="w-5 h-5 text-green-600" />
                    Masked (A')
                  </h2>
                  <button
                    onClick={() => copyToClipboard(maskedInput, 'A-comp')}
                    className="p-2 hover:bg-slate-100 rounded-lg transition-colors"
                  >
                    {copiedStage === 'A-comp' ? (
                      <Check className="w-4 h-4 text-green-600" />
                    ) : (
                      <Copy className="w-4 h-4 text-slate-500" />
                    )}
                  </button>
                </div>
                <div className="bg-green-50 rounded-lg p-4 font-mono text-sm text-slate-700 whitespace-pre-wrap border-2 border-green-200 min-h-32">
                  {maskedInput}
                </div>
              </div>
            </div>

            {/* Right Column - Output */}
            <div className="space-y-6">
              <div className="bg-white rounded-xl shadow-lg p-6">
                <div className="flex items-center justify-between mb-4">
                  <h2 className="text-lg font-semibold text-slate-800 flex items-center gap-2">
                    <Shield className="w-5 h-5 text-purple-600" />
                    Masked (B')
                  </h2>
                  <button
                    onClick={() => copyToClipboard(maskedOutput, 'B-comp')}
                    className="p-2 hover:bg-slate-100 rounded-lg transition-colors"
                  >
                    {copiedStage === 'B-comp' ? (
                      <Check className="w-4 h-4 text-green-600" />
                    ) : (
                      <Copy className="w-4 h-4 text-slate-500" />
                    )}
                  </button>
                </div>
                <div className="bg-purple-50 rounded-lg p-4 font-mono text-sm text-slate-700 whitespace-pre-wrap border-2 border-purple-200 min-h-32">
                  {maskedOutput}
                </div>
              </div>

              <div className="bg-white rounded-xl shadow-lg p-6">
                <div className="flex items-center justify-between mb-4">
                  <h2 className="text-lg font-semibold text-slate-800 flex items-center gap-2">
                    <Eye className="w-5 h-5 text-blue-600" />
                    Final (B)
                  </h2>
                  <button
                    onClick={() => copyToClipboard(finalOutput, 'final-comp')}
                    className="p-2 hover:bg-slate-100 rounded-lg transition-colors"
                  >
                    {copiedStage === 'final-comp' ? (
                      <Check className="w-4 h-4 text-green-600" />
                    ) : (
                      <Copy className="w-4 h-4 text-slate-500" />
                    )}
                  </button>
                </div>
                <div className="bg-blue-50 rounded-lg p-4 font-mono text-sm text-slate-700 whitespace-pre-wrap border-2 border-blue-200 min-h-32">
                  {finalOutput}
                </div>
              </div>
            </div>
          </div>
        )}

        {/* Diff View */}
        {activeView === 'diff' && maskedInput && (
          <div className="space-y-6">
            {/* Input Diff */}
            <div className="bg-white rounded-xl shadow-lg p-6">
              <h2 className="text-xl font-semibold text-slate-800 mb-4 flex items-center gap-2">
                <AlertCircle className="w-5 h-5 text-amber-600" />
                Input Transformation (A → A')
              </h2>
              <div className="grid md:grid-cols-2 gap-4">
                <div>
                  <div className="text-sm font-medium text-slate-600 mb-2 flex items-center gap-2">
                    <span>Original (A)</span>
                    <span className="px-2 py-0.5 bg-red-100 text-red-700 text-xs rounded">PII Exposed</span>
                  </div>
                  <div className="bg-slate-50 rounded-lg p-4 font-mono text-sm border-2 border-slate-200 whitespace-pre-wrap">
                    {inputData.split('').map((char, idx) => {
                      const isPartOfEntity = detectedEntities.some(e =>
                        inputData.indexOf(e.original) <= idx &&
                        idx < inputData.indexOf(e.original) + e.original.length
                      );
                      return (
                        <span
                          key={idx}
                          className={isPartOfEntity ? 'bg-red-200 text-red-900' : ''}
                        >
                          {char}
                        </span>
                      );
                    })}
                  </div>
                </div>
                <div>
                  <div className="text-sm font-medium text-slate-600 mb-2 flex items-center gap-2">
                    <span>Masked (A')</span>
                    <span className="px-2 py-0.5 bg-green-100 text-green-700 text-xs rounded">PII Protected</span>
                  </div>
                  <div className="bg-slate-50 rounded-lg p-4 font-mono text-sm border-2 border-slate-200 whitespace-pre-wrap">
                    {(() => {
                      let highlighted = maskedInput;
                      detectedEntities.forEach(entity => {
                        highlighted = highlighted.replace(
                          entity.token,
                          `<mark class="bg-green-200 text-green-900 font-bold">${entity.token}</mark>`
                        );
                      });
                      return <div dangerouslySetInnerHTML={{ __html: highlighted }} />;
                    })()}
                  </div>
                </div>
              </div>
              <div className="mt-4 p-3 bg-amber-50 border border-amber-200 rounded-lg">
                <p className="text-sm text-amber-900">
                  <span className="font-semibold">Changes:</span> {detectedEntities.length} PII entities detected and replaced with tokens
                </p>
              </div>
            </div>

            {/* Output Diff */}
            <div className="bg-white rounded-xl shadow-lg p-6">
              <h2 className="text-xl font-semibold text-slate-800 mb-4 flex items-center gap-2">
                <AlertCircle className="w-5 h-5 text-blue-600" />
                Output Transformation (B' → B)
              </h2>
              <div className="grid md:grid-cols-2 gap-4">
                <div>
                  <div className="text-sm font-medium text-slate-600 mb-2 flex items-center gap-2">
                    <span>Masked Output (B')</span>
                    <span className="px-2 py-0.5 bg-purple-100 text-purple-700 text-xs rounded">From OpenAI</span>
                  </div>
                  <div className="bg-slate-50 rounded-lg p-4 font-mono text-sm border-2 border-slate-200 whitespace-pre-wrap">
                    {(() => {
                      let highlighted = maskedOutput;
                      detectedEntities.forEach(entity => {
                        highlighted = highlighted.replace(
                          entity.token,
                          `<mark class="bg-purple-200 text-purple-900 font-bold">${entity.token}</mark>`
                        );
                      });
                      return <div dangerouslySetInnerHTML={{ __html: highlighted }} />;
                    })()}
                  </div>
                </div>
                <div>
                  <div className="text-sm font-medium text-slate-600 mb-2 flex items-center gap-2">
                    <span>Final Output (B)</span>
                    <span className="px-2 py-0.5 bg-blue-100 text-blue-700 text-xs rounded">Restored</span>
                  </div>
                  <div className="bg-slate-50 rounded-lg p-4 font-mono text-sm border-2 border-slate-200 whitespace-pre-wrap">
                    {(() => {
                      let highlighted = finalOutput;
                      detectedEntities.forEach(entity => {
                        highlighted = highlighted.replace(
                          entity.original,
                          `<mark class="bg-blue-200 text-blue-900 font-bold">${entity.original}</mark>`
                        );
                      });
                      return <div dangerouslySetInnerHTML={{ __html: highlighted }} />;
                    })()}
                  </div>
                </div>
              </div>
              <div className="mt-4 p-3 bg-blue-50 border border-blue-200 rounded-lg">
                <p className="text-sm text-blue-900">
                  <span className="font-semibold">Changes:</span> {detectedEntities.length} tokens replaced with original PII values
                </p>
              </div>
            </div>

            {/* Transformation Summary */}
            <div className="bg-gradient-to-r from-slate-50 to-slate-100 rounded-xl shadow-lg p-6">
              <h3 className="text-lg font-semibold text-slate-800 mb-4">Transformation Summary</h3>
              <div className="grid md:grid-cols-3 gap-4">
                <div className="bg-white rounded-lg p-4 border-l-4 border-amber-500">
                  <div className="text-2xl font-bold text-slate-800">{detectedEntities.length}</div>
                  <div className="text-sm text-slate-600">Entities Detected</div>
                </div>
                <div className="bg-white rounded-lg p-4 border-l-4 border-green-500">
                  <div className="text-2xl font-bold text-slate-800">100%</div>
                  <div className="text-sm text-slate-600">PII Protected</div>
                </div>
                <div className="bg-white rounded-lg p-4 border-l-4 border-blue-500">
                  <div className="text-2xl font-bold text-slate-800">
                    {detectedEntities.length > 0
                      ? ((detectedEntities.reduce((sum, e) => sum + e.confidence, 0) / detectedEntities.length) * 100).toFixed(1)
                      : 0}%
                  </div>
                  <div className="text-sm text-slate-600">Avg. Confidence</div>
                </div>
              </div>
            </div>
          </div>
        )}

        {/* Info Footer */}
        <div className="mt-8 text-center text-sm text-slate-500">
          <p>Privacy Proxy Service UI - Connected to real PII detection and masking pipeline.</p>
        </div>
      </div>
    </div>
  );
}
