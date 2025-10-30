import React, { useState, useEffect, useRef } from 'react';
import { Book } from '../types.ts';
import { renderPdfFirstPageToDataUrl } from '../services/pdfService.ts';
import StarRating from './StarRating.tsx';


// A more visually engaging BookCover component for the grid view
const BookCover: React.FC<{ book: Book; onZoomRequest: (imageUrl: string) => void; }> = ({ book, onZoomRequest }) => {
    const [imageSrc, setImageSrc] = useState(book.imageUrl);
    const [isLoading, setIsLoading] = useState(true);
    // Use a ref to track if we've already tried the PDF fallback to prevent loops.
    const pdfFallbackAttempted = useRef(false);

    // Reset state when the book prop (specifically its URL) changes.
    useEffect(() => {
        setImageSrc(book.imageUrl);
        setIsLoading(true);
        pdfFallbackAttempted.current = false;
    }, [book.imageUrl]);

    const handleImageError = () => {
        // First try the PDF fallback if available and not yet attempted.
        if (book.contentPath && book.contentPath.endsWith('.pdf') && !pdfFallbackAttempted.current) {
            pdfFallbackAttempted.current = true; // Mark that we are trying it now.
            renderPdfFirstPageToDataUrl(book.contentPath)
                .then(dataUrl => {
                    setImageSrc(dataUrl); // This will re-trigger the render and the img tag will try loading the new src.
                })
                .catch(e => {
                    console.error("Failed to generate PDF cover on error:", e);
                    // If PDF render fails, go to the final fallback.
                    setImageSrc('https://images.unsplash.com/photo-1589998059171-988d887df646?q=80&w=400&auto-format&fit=crop');
                });
        } else {
            // If no PDF fallback or it has already been tried, use the final static fallback.
            setImageSrc('https://images.unsplash.com/photo-1589998059171-988d887df646?q=80&w=400&auto-format&fit=crop');
        }
    };
    
    // The final source for the img tag is whatever is currently in the state.
    const finalSrc = imageSrc || 'https://images.unsplash.com/photo-1589998059171-988d887df646?q=80&w=400&auto-format&fit=crop';
    
    return (
         <div 
            className="w-full h-80 block overflow-hidden bg-slate-200 relative"
        >
            {isLoading && (
                 <div className="absolute inset-0 w-full h-full bg-slate-200 animate-pulse"></div>
            )}
             <button 
                onClick={() => onZoomRequest(finalSrc)}
                // Use opacity to hide/show for a smoother transition than display:none
                className={`w-full h-full transition-opacity duration-500 ${isLoading ? 'opacity-0' : 'opacity-100'}`}
                aria-label={`عرض غلاف كتاب ${book.title}`}
            >
                <img 
                    loading="lazy"
                    src={finalSrc} 
                    alt={`غلاف كتاب ${book.title}`}
                    className="h-full w-full object-cover group-hover:scale-105 transition-transform duration-300 ease-in-out"
                    onLoad={() => setIsLoading(false)}
                    onError={handleImageError}
                />
            </button>
        </div>
    );
};

interface SelectorProps {
    title: string;
    addButtonText: string;
    searchPlaceholder: string;
    books: Book[];
    onAnalyze: (book: Book) => void;
    isAnalyzing: boolean;
    analyzingItemId: string | null;
    onOpenAddModal: () => void;
    onOpenNotesListModal: () => void;
    editingId: string | null;
    onStartEdit: (bookId: string | null) => void;
    onSaveEdit: (bookId: string, newTitle: string) => void;
    editingDescId: string | null;
    onStartDescEdit: (bookId: string | null) => void;
    onSaveDescEdit: (bookId: string, newDescription: string) => void;
    onDelete: (book: Book) => void;
    onRate: (bookId: string, newRating: number) => void;
    initialItemIds: Set<string>;
}

const BookSelector: React.FC<SelectorProps> = ({ 
    title,
    addButtonText,
    searchPlaceholder,
    books, 
    onAnalyze, 
    isAnalyzing, 
    analyzingItemId,
    onOpenAddModal,
    onOpenNotesListModal,
    editingId,
    onStartEdit,
    onSaveEdit,
    editingDescId,
    onStartDescEdit,
    onSaveDescEdit,
    onDelete,
    onRate,
    initialItemIds,
}) => {
    const [zoomedImage, setZoomedImage] = useState<string | null>(null);
    const [isHelpModalOpen, setIsHelpModalOpen] = useState(false);
    const [searchQuery, setSearchQuery] = useState('');
    const [activeCategories, setActiveCategories] = useState<Set<string>>(new Set());

    const allCategories = [...new Set(books.flatMap(book => book.categories || []))].sort();

    const handleCategoryToggle = (category: string) => {
        setActiveCategories(prev => {
            const newSet = new Set(prev);
            if (newSet.has(category)) {
                newSet.delete(category);
            } else {
                newSet.add(category);
            }
            return newSet;
        });
    };

    const handleTitleKeyDown = (e: React.KeyboardEvent<HTMLInputElement>, bookId: string) => {
        if (e.key === 'Enter') {
            // FIX: Explicitly cast event target to resolve type inference issue and safely access its value.
            onSaveEdit(bookId, (e.target as HTMLInputElement).value);
        } else if (e.key === 'Escape') {
            onStartEdit(null);
        }
    };

    const handleDescriptionKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>, bookId: string) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            // FIX: Explicitly cast event target to resolve type inference issue and safely access its value.
            onSaveDescEdit(bookId, (e.target as HTMLTextAreaElement).value);
        } else if (e.key === 'Escape') {
            onStartDescEdit(null);
        }
    };

    const filteredBooks = books.filter(book => {
        const matchesSearch = book.title.toLowerCase().includes(searchQuery.toLowerCase()) ||
                              (book.description && book.description.toLowerCase().includes(searchQuery.toLowerCase()));
        
        const matchesCategory = activeCategories.size === 0 || 
                                (book.categories && [...activeCategories].every(cat => book.categories!.includes(cat)));

        return matchesSearch && matchesCategory;
    });

    return (
        <div className="p-6 bg-white rounded-2xl shadow-lg border border-slate-200">
            <div className="flex justify-between items-center mb-4 border-b pb-3 border-slate-200 flex-wrap gap-2">
                <div className="text-2xl font-bold text-slate-800">
                    {title}
                </div>
                <div className="flex items-center gap-3">
                    <button onClick={() => setIsHelpModalOpen(true)} className="p-2 text-slate-500 rounded-full hover:bg-slate-100 hover:text-slate-800 transition-colors duration-200" aria-label="تعليمات" data-tooltip="عرض التعليمات">
                        <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth="2"><path strokeLinecap="round" strokeLinejoin="round" d="M9.879 7.519c1.171-1.025 3.071-1.025 4.242 0 1.172 1.025 1.172 2.687 0 3.712-.203.179-.43.326-.67.442-.745.361-1.45.999-1.45 1.827v.75M21 12a9 9 0 11-18 0 9 9 0 0118 0zm-9 5.25h.008v.008H12v-.008z" /></svg>
                    </button>
                    <button onClick={onOpenNotesListModal} className="flex items-center gap-2 bg-white text-slate-700 font-semibold py-2 px-4 rounded-lg hover:bg-slate-50 transition-colors duration-200 text-sm shadow-sm border border-slate-300">
                        <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth="2"><path strokeLinecap="round" strokeLinejoin="round" d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" /></svg>
                        <span>مذكراتي</span>
                    </button>
                    <button onClick={onOpenAddModal} className="flex items-center gap-2 bg-blue-600 text-white font-semibold py-2 px-4 rounded-lg hover:bg-blue-700 transition-colors duration-200 text-sm shadow hover:shadow-md">
                        <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5" viewBox="0 0 20 20" fill="currentColor"><path fillRule="evenodd" d="M10 5a1 1 0 011 1v3h3a1 1 0 110 2h-3v3a1 1 0 11-2 0v-3H6a1 1 0 110 2h3V6a1 1 0 011-1z" clipRule="evenodd" /></svg>
                        <span>{addButtonText}</span>
                    </button>
                </div>
            </div>
            
            <div className="my-4 relative">
                <input type="search" placeholder={searchPlaceholder} value={searchQuery} onChange={(e) => setSearchQuery(e.target.value)} className="w-full pl-10 pr-4 py-2.5 border border-slate-300 rounded-lg focus:ring-2 focus:ring-blue-400 focus:border-blue-400 transition-shadow" aria-label="ابحث في المكتبة"/>
                <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none"><svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5 text-slate-400" viewBox="0 0 20 20" fill="currentColor"><path fillRule="evenodd" d="M8 4a4 4 0 100 8 4 4 0 000-8zM2 8a6 6 0 1110.89 3.476l4.817 4.817a1 1 0 01-1.414 1.414l-4.816-4.816A6 6 0 012 8z" clipRule="evenodd" /></svg></div>
            </div>
            
             {allCategories.length > 0 && (
                <div className="mb-6 pt-4 border-t border-slate-200">
                    <div className="flex items-center flex-wrap gap-2">
                        <span className="font-semibold text-slate-600 text-sm ml-2">فلترة حسب التصنيف:</span>
                        <button
                            onClick={() => setActiveCategories(new Set())}
                            className={`px-3 py-1 text-xs font-medium rounded-full border transition-colors ${activeCategories.size === 0 ? 'bg-blue-600 text-white border-blue-600' : 'bg-slate-100 text-slate-700 border-slate-200 hover:bg-slate-200'}`}
                        >
                            الكل
                        </button>
                        {allCategories.map(cat => (
                            <button 
                                key={cat}
                                onClick={() => handleCategoryToggle(cat)}
                                className={`px-3 py-1 text-xs font-medium rounded-full border transition-colors ${activeCategories.has(cat) ? 'bg-blue-600 text-white border-blue-600' : 'bg-slate-100 text-slate-700 border-slate-200 hover:bg-slate-200'}`}
                            >
                                {cat}
                            </button>
                        ))}
                    </div>
                </div>
            )}


            <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-8 pt-4">
                {filteredBooks.length > 0 ? (
                    filteredBooks.map((book) => {
                        if (book.isLoading) {
                            return (
                                <div key={book.id} className="group flex flex-col bg-slate-100 rounded-xl shadow-md overflow-hidden border border-slate-200">
                                    <div className="w-full h-80 block overflow-hidden bg-slate-200 animate-pulse relative flex flex-col items-center justify-center">
                                        <svg className="animate-spin h-10 w-10 text-slate-400" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                                            <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                                            <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                                        </svg>
                                        <p className="text-slate-500 font-semibold mt-4">جاري الإضافة...</p>
                                    </div>
                                    <div className="p-5 flex flex-col flex-grow">
                                        <div className="h-6 bg-slate-200 rounded w-3/4 mb-2 animate-pulse"></div>
                                        <div className="flex-grow min-h-[60px] space-y-2">
                                            <div className="h-4 bg-slate-200 rounded w-full animate-pulse"></div>
                                            <div className="h-4 bg-slate-200 rounded w-5/6 animate-pulse"></div>
                                        </div>
                                        <div className="mt-auto pt-4 border-t border-slate-200 space-y-3">
                                            <div className="flex items-center gap-2">
                                                <div className="flex-1 h-9 bg-slate-200 rounded-lg animate-pulse"></div>
                                                <div className="flex-1 h-9 bg-slate-200 rounded-lg animate-pulse"></div>
                                            </div>
                                            <div className="w-full h-10 bg-slate-300 rounded-lg animate-pulse"></div>
                                        </div>
                                    </div>
                                </div>
                            );
                        }

                        const isOriginalCoreBook = initialItemIds.has(book.id);
                        const wasAnalyzed = !!book.isUserAdded;
                        const analyzeButtonText = (isOriginalCoreBook && wasAnalyzed) ? "عرض التحليل" : "تحليل";
                        const isCurrentlyAnalyzing = analyzingItemId === book.id;
                        const isReadable = !!book.contentPath || !!book.file;

                        return (
                            <div key={book.id} className="group flex flex-col bg-slate-50 rounded-xl shadow-md hover:shadow-2xl transition-all duration-300 ease-in-out overflow-hidden border border-slate-200 hover:border-blue-400">
                                <div className="relative">
                                    <BookCover book={book} onZoomRequest={setZoomedImage} />
                                    <div className="absolute top-3 right-3 flex items-center space-x-1 opacity-0 group-hover:opacity-100 transition-opacity duration-300 bg-black bg-opacity-40 p-1 rounded-lg">
                                        <button onClick={() => editingId === book.id ? onStartEdit(null) : onStartEdit(book.id)} data-tooltip="تعديل العنوان" className="text-white hover:text-blue-300 p-1.5"><svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5" viewBox="0 0 20 20" fill="currentColor"><path d="M17.414 2.586a2 2 0 00-2.828 0L7 10.172V13h2.828l7.586-7.586a2 2 0 000-2.828z" /><path fillRule="evenodd" d="M2 6a2 2 0 012-2h4a1 1 0 010 2H4v10h10v-4a1 1 0 112 0v4a2 2 0 01-2 2H4a2 2 0 01-2-2V6z" clipRule="evenodd" /></svg></button>
                                        <button onClick={() => onDelete(book)} data-tooltip="حذف" className="text-white hover:text-red-400 p-1.5"><svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5" viewBox="0 0 20 20" fill="currentColor"><path fillRule="evenodd" d="M9 2a1 1 0 00-.894.553L7.382 4H4a1 1 0 000 2v10a2 2 0 002 2h8a2 2 0 002-2V6a1 1 0 100-2h-3.382l-.724-1.447A1 1 0 0011 2H9zM7 8a1 1 0 012 0v6a1 1 0 11-2 0V8zm4 0a1 1 0 012 0v6a1 1 0 11-2 0V8z" clipRule="evenodd" /></svg></button>
                                    </div>
                                </div>
                                
                                <div className="p-5 flex flex-col flex-grow">
                                     {editingId === book.id ? (
                                        <input type="text" defaultValue={book.title} onKeyDown={(e) => handleTitleKeyDown(e, book.id)} onBlur={(e) => onSaveEdit(book.id, (e.target as HTMLInputElement).value)} className="font-bold text-lg text-slate-800 bg-white border border-blue-400 rounded px-2 py-1 w-full mb-2" autoFocus />
                                    ) : (
                                        <h3 className="font-bold text-lg text-slate-800 group-hover:text-blue-600 transition-colors duration-300 mb-2 truncate" title={book.title}>{book.title}</h3>
                                    )}
        
                                    <div className="flex-grow min-h-[60px]">
                                         {editingDescId === book.id ? (
                                            <textarea defaultValue={book.description} onKeyDown={(e) => handleDescriptionKeyDown(e, book.id)} onBlur={(e) => onSaveDescEdit(book.id, (e.target as HTMLTextAreaElement).value)} className="text-sm text-slate-600 bg-white border border-blue-400 rounded px-2 py-1 w-full" autoFocus rows={4} />
                                        ) : (
                                            <div className="flex items-start gap-1">
                                                <p className="text-sm text-slate-600 line-clamp-2 flex-grow">{book.description}</p>
                                                <button onClick={() => editingDescId === book.id ? onStartDescEdit(null) : onStartDescEdit(book.id)} data-tooltip="تعديل الوصف" className="text-slate-400 hover:text-blue-600 p-1 opacity-0 group-hover:opacity-100 transition-opacity flex-shrink-0"><svg xmlns="http://www.w3.org/2000/svg" className="h-4 w-4" viewBox="0 0 20 20" fill="currentColor"><path d="M17.414 2.586a2 2 0 00-2.828 0L7 10.172V13h2.828l7.586-7.586a2 2 0 000-2.828z" /><path fillRule="evenodd" d="M2 6a2 2 0 012-2h4a1 1 0 010 2H4v10h10v-4a1 1 0 112 0v4a2 2 0 01-2 2H4a2 2 0 01-2-2V6z" clipRule="evenodd" /></svg></button>
                                            </div>
                                         )}
                                    </div>
                                    
                                    {book.categories && book.categories.length > 0 && (
                                        <div className="flex flex-wrap gap-2 my-3">
                                            {book.categories.map(cat => (
                                                <span key={cat} className="bg-slate-200 text-slate-700 text-xs font-semibold px-2.5 py-1 rounded-full">
                                                    {cat}
                                                </span>
                                            ))}
                                        </div>
                                    )}

                                    {book.communityRating && book.ratingCount && (
                                        <div className="text-xs text-slate-500 mt-3 mb-1 flex items-center gap-2" dir="rtl">
                                            <span className="font-semibold">التقييم العام:</span>
                                            <div className="flex items-center gap-2"><StarRating rating={book.communityRating} size="sm" /><span className="font-bold text-slate-600">{book.communityRating.toFixed(1)}</span></div>
                                            <span className="text-slate-400">({new Intl.NumberFormat('ar-EG').format(book.ratingCount)} مشارك)</span>
                                        </div>
                                    )}

                                    <div className="flex items-center justify-between my-3">
                                        <StarRating rating={book.rating || 0} onRate={(newRating) => onRate(book.id, newRating)} />
                                        {book.rating ? <span className="text-xs font-bold text-slate-500 bg-slate-200 px-2 py-0.5 rounded-full">تقييمك: {book.rating.toFixed(1)}</span> : <span className="text-xs text-slate-400">قيّم بنفسك</span>}
                                    </div>
                                    
                                    <div className="mt-auto pt-4 border-t border-slate-200 space-y-3">
                                        <div className="flex items-center gap-2">
                                            <a 
                                                href={isReadable ? (book.downloadUrl || book.contentPath) : undefined} 
                                                target="_blank" 
                                                rel="noopener noreferrer" 
                                                onClick={(e) => !isReadable && e.preventDefault()}
                                                className={`flex-1 flex items-center justify-center text-center border border-sky-600 text-sky-600 font-semibold py-2 px-3 rounded-lg hover:bg-sky-50 transition-colors duration-300 text-sm ${!isReadable ? 'opacity-50 cursor-not-allowed' : ''}`}
                                            >
                                                <svg xmlns="http://www.w3.org/2000/svg" className="h-4 w-4" viewBox="0 0 20 20" fill="currentColor"><path d="M10 12a2 2 0 100-4 2 2 0 000 4z" /><path fillRule="evenodd" d="M.458 10C1.732 5.943 5.522 3 10 3s8.268 2.943 9.542 7c-1.274 4.057-5.022 7-9.542 7S1.732 14.057.458 10zM14 10a4 4 0 11-8 0 4 4 0 018 0z" clipRule="evenodd" /></svg>
                                                <span className="mr-2">قراءة</span>
                                            </a>
                                            <a href={book.downloadActionUrl || book.downloadUrl || '#'} target="_blank" rel="noopener noreferrer" download onClick={(e) => !(book.downloadActionUrl || book.downloadUrl) && e.preventDefault()} className={`flex-1 flex items-center justify-center text-center border border-green-600 text-green-600 font-semibold py-2 px-3 rounded-lg hover:bg-green-50 transition-colors duration-300 text-sm ${!(book.downloadActionUrl || book.downloadUrl) ? 'opacity-50 cursor-not-allowed' : ''}`}>
                                                <svg xmlns="http://www.w3.org/2000/svg" className="h-4 w-4" viewBox="0 0 20 20" fill="currentColor"><path fillRule="evenodd" d="M3 17a1 1 0 011-1h12a1 1 0 110 2H4a1 1 0 01-1-1zm3.293-7.707a1 1 0 011.414 0L9 10.586V3a1 1 0 112 0v7.586l1.293-1.293a1 1 0 111.414 1.414l-3 3a1 1 0 01-1.414 0l-3-3a1 1 0 010-1.414z" clipRule="evenodd" /></svg>
                                                <span className="mr-2">تحميل</span>
                                            </a>
                                        </div>
                                        <button onClick={() => onAnalyze(book)} disabled={isAnalyzing || !isReadable} className="w-full bg-blue-600 text-white font-semibold py-2.5 px-4 rounded-lg hover:bg-blue-700 disabled:bg-slate-400 disabled:cursor-not-allowed transition-colors duration-300 shadow hover:shadow-lg flex justify-center items-center">
                                            {isCurrentlyAnalyzing ? (<> <svg className="animate-spin -ml-1 mr-3 h-5 w-5 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24"><circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle><path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path></svg> <span>جارٍ التحليل...</span> </>) 
                                            : (<> <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth="2"><path strokeLinecap="round" strokeLinejoin="round" d="M5 3v4M3 5h4M6 17v4m-2-2h4m5-16l2.286 6.857L21 12l-5.714 2.143L13 21l-2.286-6.857L5 12l5.714-2.143L13 3z" /></svg> <span className="mr-2">{analyzeButtonText}</span></>)}
                                        </button>
                                    </div>
                                </div>
                            </div>
                        );
                    })
                ) : (
                    <div className="col-span-full text-center py-12 text-slate-500">
                        <h3 className="text-xl font-semibold text-slate-700">لم يتم العثور على نتائج</h3>
                        <p className="mt-2">حاول تعديل مصطلح البحث أو الفلترة.</p>
                    </div>
                )}
            </div>

            {zoomedImage && (
                <div className="fixed inset-0 bg-black bg-opacity-80 flex justify-center items-center z-50 p-4 transition-opacity duration-300 animate-fade-in" onClick={() => setZoomedImage(null)} style={{ animation: 'fadeIn 0.2s ease-out' }}>
                    <div className="relative max-w-[90vw] max-h-[90vh] flex" onClick={(e) => e.stopPropagation()}>
                        <img src={zoomedImage} alt="غلاف مكبر" className="object-contain rounded-lg shadow-2xl" />
                        <button onClick={() => setZoomedImage(null)} className="absolute -top-3 -right-3 bg-white text-slate-800 rounded-full h-9 w-9 flex items-center justify-center text-2xl font-bold shadow-lg hover:bg-slate-200 transition-transform hover:scale-110" aria-label="إغلاق">&times;</button>
                    </div>
                     <style>{`@keyframes fadeIn { from { opacity: 0; } to { opacity: 1; } }`}</style>
                </div>
            )}
            
            {isHelpModalOpen && (
                <div className="fixed inset-0 bg-black bg-opacity-60 flex justify-center items-center z-50" onClick={() => setIsHelpModalOpen(false)}>
                    <div className="bg-white p-8 rounded-2xl shadow-2xl w-full max-w-md m-4 text-center" onClick={e => e.stopPropagation()}>
                        <h2 className="text-2xl font-bold text-slate-800 mb-4">تعليمات</h2>
                        <p className="text-slate-600 mb-8">تستطيع إضافة كتبك الخاصة ولن تظهر لأحد غيرك ولو أردتها تظهر للآخرين أرسلها لنا لنضيفها (بشرط أن تكون متناسبة مع برنامجنا) علما أن كتبك ستظهر لك إلا من خلال جهازك فلو مسحت بيانات متصفحك مثلاً ستذهب كلها كما يمكنك مسح كتاب معين من أيقونة المسح التي تظهر على صورة الكتاب .</p>
                        <div className="flex justify-center">
                            <button onClick={() => setIsHelpModalOpen(false)} className="px-6 py-2 bg-blue-600 text-white font-semibold rounded-lg hover:bg-blue-700">إغلاق</button>
                        </div>
                    </div>
                </div>
            )}
        </div>
    );
};

export default BookSelector;