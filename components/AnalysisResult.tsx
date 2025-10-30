import React, { useState, useEffect } from 'react';
import Spinner from './Spinner';
import { QuizResult, Book, Video } from '../types';

interface AnalysisResultProps {
    analysis: string;
    isLoading: boolean;
    isSummarizing?: boolean;
    analyzedItemName?: string | null;
    showFollowUpActions?: boolean;
    onSummarize?: () => void;
    onCreateQuiz?: () => void;
    onAnalyzeSentiment?: () => void;
    onExtractKeywords?: () => void;
    onTranslate?: (targetLanguage: 'en' | 'ar') => void;
    onRequestMoreQuestions?: () => void;
    onGenerateSuggestions?: () => void;
    quizResult?: QuizResult | null;
    highlightedKeywords?: string[];
    onClearHighlights?: () => void;
    isGeneratingSuggestions?: boolean;
    suggestedContent?: (Book | Video)[];
    onGoToLibrary?: () => void;
    onDesignArticle?: () => void;
    isDesigningArticle?: boolean;
    onRateContent?: () => void;
    isRating?: boolean;
    isAnalyzedItemUserAdded?: boolean;
}

const highlightText = (text: string, keywords: string[]): React.ReactNode => {
    if (!keywords || keywords.length === 0 || !text) {
        return text;
    }
    const validKeywords = keywords.filter(kw => kw && kw.trim() !== '');
    if (validKeywords.length === 0) return text;

    const escapedKeywords = validKeywords.map(kw => kw.replace(/[-\/\\^$*+?.()|[\]{}]/g, '\\$&'));
    
    // Removed word boundaries (\b) to correctly highlight words in Arabic, which can be connected.
    const regex = new RegExp(`(${escapedKeywords.join('|')})`, 'gi');
    const parts = text.split(regex);

    return (
        <>
            {parts.map((part, index) => {
                // When using split with a capturing group, matched parts are at odd indices.
                // We double-check if the part is one of our keywords (case-insensitively).
                const isKeyword = index % 2 === 1 && validKeywords.some(kw => kw.toLowerCase() === part.toLowerCase());
                return isKeyword ? (
                    <mark key={index} className="bg-yellow-200 px-1 rounded-sm">
                        {part}
                    </mark>
                ) : (
                    part
                );
            })}
        </>
    );
};


const AnalysisResult: React.FC<AnalysisResultProps> = ({ 
    analysis, 
    isLoading, 
    isSummarizing,
    analyzedItemName,
    showFollowUpActions,
    onSummarize,
    onCreateQuiz,
    onAnalyzeSentiment,
    onExtractKeywords,
    onTranslate,
    onRequestMoreQuestions,
    onGenerateSuggestions,
    quizResult,
    highlightedKeywords = [],
    onClearHighlights,
    isGeneratingSuggestions,
    suggestedContent = [],
    onGoToLibrary,
    onDesignArticle,
    isDesigningArticle,
    onRateContent,
    isRating,
    isAnalyzedItemUserAdded,
}) => {
    const [isExpanded, setIsExpanded] = useState(false);
    const [targetLang, setTargetLang] = useState<'en' | 'ar'>('en');
    const areKeywordsHighlighted = highlightedKeywords.length > 0;

    useEffect(() => {
        setIsExpanded(false);
        // Detect if the text is primarily Arabic to suggest the correct translation target.
        const isArabic = /[\u0600-\u06FF]/.test(analysis);
        setTargetLang(isArabic ? 'en' : 'ar');
    }, [analysis]);

    const TRUNCATE_THRESHOLD = 1000;
    const needsTruncation = analysis.length > TRUNCATE_THRESHOLD;

    const displayedText = needsTruncation && !isExpanded 
        ? `${analysis.substring(0, TRUNCATE_THRESHOLD)}...` 
        : analysis;

    // Show full-page loader only on the initial analysis when there's no content yet.
    if (isLoading && !isSummarizing && !analysis.trim() && !quizResult) {
        return (
            <div className="p-6 bg-white rounded-2xl shadow-lg border border-slate-200 min-h-[200px] flex flex-col items-center justify-center">
                <Spinner />
                <p className="mt-4 text-slate-600 font-semibold">
                    {analyzedItemName ? `يقوم الذكاء الاصطناعي بتحليل "${analyzedItemName}"...` : 'يرجى الانتظار...'}
                </p>
            </div>
        );
    }

    // Display quiz result if it exists
    if (quizResult) {
        return (
             <div className="p-6 bg-white rounded-2xl shadow-lg border border-slate-200 text-center">
                <h3 className="text-2xl font-bold text-slate-800 mb-4">نتيجة الاختبار لكتاب: {analyzedItemName}</h3>
                <p className="text-4xl font-bold text-blue-600 my-4">
                    {quizResult.score} / {quizResult.total}
                </p>
                <p className="text-lg text-slate-600 mb-6">
                    {quizResult.score === quizResult.total ? 'ممتاز! نتيجة كاملة!' : 'أحسنت! حاول مرة أخرى لتحسين نتيجتك.'}
                </p>
                {showFollowUpActions && (
                    <div className="mt-6 pt-4 border-t border-slate-200 flex flex-col sm:flex-row gap-4">
                        <button
                            onClick={onSummarize}
                            disabled={isLoading}
                            className="flex-1 bg-slate-100 text-slate-700 font-semibold py-2 px-4 rounded-lg hover:bg-slate-200 transition-colors duration-300 border border-slate-300 disabled:opacity-50 disabled:cursor-wait"
                        >
                            عرض التحليل الأصلي
                        </button>
                        <button
                            onClick={onRequestMoreQuestions}
                            disabled={isLoading}
                            className="flex-1 bg-blue-100 text-blue-700 font-semibold py-2 px-4 rounded-lg hover:bg-blue-200 transition-colors duration-300 border border-blue-300 disabled:opacity-50 disabled:cursor-wait"
                        >
                            تريد أسئلة أكثر؟
                        </button>
                    </div>
                )}
            </div>
        );
    }
    
    if (!analysis && !isSummarizing && !isDesigningArticle && !isRating) {
        return null;
    }

    return (
        <div className="p-6 bg-white rounded-2xl shadow-lg border border-slate-200">
             {isSummarizing && (
                <div className="mb-4">
                    <h4 className="text-lg font-semibold text-slate-700 mb-2">يقوم الذكاء الاصطناعي بالتلخيص المفصل...</h4>
                    <div className="w-full bg-slate-200 rounded-full h-2.5 overflow-hidden">
                        <div className="bg-blue-600 h-2.5 w-full rounded-full animate-indeterminate-progress"></div>
                    </div>
                </div>
            )}
            {isDesigningArticle && (
                <div className="mb-4">
                    <h4 className="text-lg font-semibold text-slate-700 mb-2">يقوم الذكاء الاصطناعي بتصميم المقال...</h4>
                    <div className="w-full bg-slate-200 rounded-full h-2.5 overflow-hidden">
                        <div className="bg-blue-600 h-2.5 w-full rounded-full animate-indeterminate-progress"></div>
                    </div>
                </div>
            )}
             {isRating && (
                <div className="mb-4">
                    <h4 className="text-lg font-semibold text-slate-700 mb-2">يقوم الذكاء الاصطناعي بتقييم المحتوى...</h4>
                    <div className="w-full bg-slate-200 rounded-full h-2.5 overflow-hidden">
                        <div className="bg-blue-600 h-2.5 w-full rounded-full animate-indeterminate-progress"></div>
                    </div>
                </div>
            )}
            <h3 className="text-2xl font-bold text-slate-800 mb-4 border-b pb-2 border-slate-200">
                {analyzedItemName ? `نتائج تحليل: ${analyzedItemName}` : 'نتائج التحليل'}
            </h3>
            <div className="prose prose-slate max-w-none whitespace-pre-wrap">
                {highlightText(displayedText, highlightedKeywords)}
            </div>

            {needsTruncation && (
                <button
                    onClick={() => setIsExpanded(!isExpanded)}
                    className="mt-4 text-blue-600 font-semibold hover:underline"
                >
                    {isExpanded ? 'عرض أقل' : 'قراءة المزيد'}
                </button>
            )}

            {showFollowUpActions && (
                 <div className="mt-6 pt-4 border-t border-slate-200">
                    <h4 className="text-lg font-semibold text-slate-700 mb-4 text-center">ماذا تريد أن تفعل الآن؟</h4>
                    <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-3">
                        <button
                            onClick={onSummarize}
                            disabled={isLoading || isSummarizing || isDesigningArticle || isRating}
                            data-tooltip="يقوم الذكاء الاصطناعي بقراءة النص فقرة بفقرة وتقديم شرح مفصل لكل منها."
                            className="w-full bg-slate-100 text-slate-700 font-semibold py-2.5 px-4 rounded-lg hover:bg-slate-200 transition-colors duration-300 border border-slate-300 disabled:opacity-50 disabled:cursor-wait"
                        >
                            تلخيص مفصل
                        </button>
                        <button
                            onClick={onCreateQuiz}
                            disabled={isLoading || isSummarizing || isDesigningArticle || isRating}
                            data-tooltip="ينشئ اختبار قصير من عدة أسئلة لاختبار فهمك لمحتوى الكتاب."
                            className="w-full bg-slate-100 text-slate-700 font-semibold py-2.5 px-4 rounded-lg hover:bg-slate-200 transition-colors duration-300 border border-slate-300 disabled:opacity-50 disabled:cursor-wait"
                        >
                            إنشاء اختبار
                        </button>
                        <button
                            onClick={onDesignArticle}
                            disabled={isLoading || isSummarizing || isDesigningArticle || isGeneratingSuggestions || isRating}
                            data-tooltip="يحول تحليل الكتاب إلى مقال جاهز للنشر في مدونتك الشخصية."
                            className="w-full bg-slate-100 text-slate-700 font-semibold py-2.5 px-4 rounded-lg hover:bg-slate-200 transition-colors duration-300 border border-slate-300 disabled:opacity-50 disabled:cursor-wait flex items-center justify-center"
                        >
                            {isDesigningArticle ? <><Spinner /> <span className="mr-2">جاري التصميم...</span></> : 'صمم مقالا'}
                        </button>
                        <button
                            onClick={onRateContent}
                            disabled={isLoading || isSummarizing || isDesigningArticle || isGeneratingSuggestions || isRating || !isAnalyzedItemUserAdded}
                            data-tooltip={!isAnalyzedItemUserAdded ? "التقييم متاح فقط للمحتوى الذي تضيفه." : "يقوم بتقييم المحتوى ويعطي درجة من 5 مع مبررات التقييم."}
                            className="w-full bg-slate-100 text-slate-700 font-semibold py-2.5 px-4 rounded-lg hover:bg-slate-200 transition-colors duration-300 border border-slate-300 disabled:opacity-50 disabled:cursor-wait flex items-center justify-center"
                        >
                            {isRating ? <><Spinner /> <span className="mr-2">جاري التقييم...</span></> : 'تقييم المحتوى'}
                        </button>
                        <button
                            onClick={onAnalyzeSentiment}
                            disabled={isLoading || isSummarizing || isDesigningArticle || isRating}
                            data-tooltip="يحدد ما إذا كان الأسلوب العام للكتاب إيجابيًا، سلبيًا، أو محايدًا مع تقديم شرح."
                            className="w-full bg-slate-100 text-slate-700 font-semibold py-2.5 px-4 rounded-lg hover:bg-slate-200 transition-colors duration-300 border border-slate-300 disabled:opacity-50 disabled:cursor-wait"
                        >
                            تحليل المشاعر
                        </button>
                        <button
                            onClick={onGenerateSuggestions}
                            disabled={isLoading || isGeneratingSuggestions || isSummarizing || isDesigningArticle || isRating}
                            data-tooltip="يقترح كتبًا أو فيديوهات ذات صلة من مكتبتك بناءً على محتوى الكتاب الحالي."
                            className="w-full bg-slate-100 text-slate-700 font-semibold py-2.5 px-4 rounded-lg hover:bg-slate-200 transition-colors duration-300 border border-slate-300 disabled:opacity-50 disabled:cursor-wait flex items-center justify-center"
                        >
                             {isGeneratingSuggestions ? <><Spinner /> <span className="mr-2">جارٍ البحث...</span></> : 'اقتراحات المحتوى'}
                        </button>
                        {areKeywordsHighlighted ? (
                            <button
                                onClick={onClearHighlights}
                                disabled={isLoading || isSummarizing || isDesigningArticle || isRating}
                                data-tooltip="يزيل تمييز الكلمات الرئيسية من النص."
                                className="w-full bg-yellow-200 text-yellow-800 font-semibold py-2.5 px-4 rounded-lg hover:bg-yellow-300 transition-colors duration-300 border border-yellow-400 disabled:opacity-50 disabled:cursor-wait"
                            >
                                إزالة تمييز الكلمات
                            </button>
                        ) : (
                            <button
                                onClick={onExtractKeywords}
                                disabled={isLoading || isSummarizing || isDesigningArticle || isRating}
                                data-tooltip="يستخرج أهم المصطلحات والمواضيع الرئيسية من الكتاب ويميزها في النص أعلاه."
                                className="w-full bg-slate-100 text-slate-700 font-semibold py-2.5 px-4 rounded-lg hover:bg-slate-200 transition-colors duration-300 border border-slate-300 disabled:opacity-50 disabled:cursor-wait"
                            >
                                تمييز الكلمات الرئيسية
                            </button>
                        )}
                        <div className="sm:col-span-2 lg:col-span-3 flex items-center gap-2 p-1.5 bg-slate-100 rounded-lg border border-slate-300" data-tooltip="ترجمة النص المعروض حاليًا إلى اللغة المحددة.">
                            <label htmlFor="translation-select" className="text-slate-700 font-semibold px-2 flex-shrink-0">
                                ترجمة:
                            </label>
                            <select
                                id="translation-select"
                                value={targetLang}
                                onChange={(e) => setTargetLang(e.target.value as 'en' | 'ar')}
                                disabled={isLoading || isSummarizing || isDesigningArticle || isRating}
                                className="block w-full px-3 py-2 bg-white border border-slate-300 rounded-md shadow-sm focus:outline-none focus:ring-blue-500 focus:border-blue-500 disabled:opacity-50"
                            >
                                <option value="en">الإنجليزية</option>
                                <option value="ar">العربية</option>
                            </select>
                            <button
                                onClick={() => onTranslate && onTranslate(targetLang)}
                                disabled={isLoading || isSummarizing || isDesigningArticle || isRating}
                                className="bg-blue-600 text-white font-semibold py-2 px-4 rounded-lg hover:bg-blue-700 transition-colors duration-300 flex-shrink-0 disabled:bg-blue-400 disabled:cursor-wait"
                            >
                                نفذ
                            </button>
                        </div>
                    </div>
                </div>
            )}
            
            {suggestedContent.length > 0 && !isLoading && (
                 <div className="mt-8 pt-6 border-t border-slate-200">
                    <h4 className="text-xl font-bold text-slate-800 mb-4">محتوى مقترح قد يعجبك</h4>
                     <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-4">
                        {suggestedContent.map(item => {
                            const isBook = 'imageUrl' in item;
                            const imageUrl = isBook ? (item as Book).imageUrl : (item as Video).thumbnailUrl;
                            return (
                                <button key={item.id} onClick={onGoToLibrary} className="group text-right bg-slate-50 rounded-lg overflow-hidden border border-slate-200 hover:shadow-lg hover:border-blue-400 transition-all duration-300">
                                    <div className="w-full h-32 bg-slate-200 overflow-hidden">
                                        <img src={imageUrl} alt={`غلاف ${item.title}`} className="w-full h-full object-cover group-hover:scale-105 transition-transform" />
                                    </div>
                                    <div className="p-3">
                                        <p className="font-semibold text-sm text-slate-700 truncate group-hover:text-blue-600">{item.title}</p>
                                        <span className={`text-xs font-medium px-2 py-0.5 rounded-full mt-1 inline-block ${isBook ? 'bg-green-100 text-green-800' : 'bg-purple-100 text-purple-800'}`}>
                                            {isBook ? 'كتاب' : 'فيديو'}
                                        </span>
                                    </div>
                                </button>
                            );
                        })}
                     </div>
                 </div>
            )}
        </div>
    );
};

export default AnalysisResult;