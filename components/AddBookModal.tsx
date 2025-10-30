import React, { useState } from 'react';

export type MediaType = 'book' | 'research' | 'periodical' | 'video' | 'audio' | 'image';

interface AddMediaModalProps {
    mediaType: MediaType;
    onClose: () => void;
    onAdd: (title: string, description: string, url: string, file: File) => void;
}

const config = {
    book: {
        title: 'إضافة كتاب جديد',
        fileLabel: 'ملف الكتاب (PDF, TXT, DOCX, EPUB)',
        accept: '.pdf,.txt,.docx,.epub,.mobi,.azw,.azw3',
        urlLabel: 'رابط صورة الغلاف (اختياري)',
        urlPlaceholder: 'إذا تركته فارغًا، سيتم إنشاء غلاف.',
    },
    research: {
        title: 'إضافة بحث جديد',
        fileLabel: 'ملف البحث (PDF, TXT, DOCX, EPUB)',
        accept: '.pdf,.txt,.docx,.epub',
        urlLabel: 'رابط صورة الغلاف (اختياري)',
        urlPlaceholder: 'إذا تركته فارغًا، سيتم إنشاء غلاف.',
    },
    periodical: {
        title: 'إضافة دورية جديدة',
        fileLabel: 'ملف الدورية (PDF, TXT, DOCX, EPUB)',
        accept: '.pdf,.txt,.docx,.epub',
        urlLabel: 'رابط صورة الغلاف (اختياري)',
        urlPlaceholder: 'إذا تركته فارغًا، سيتم إنشاء غلاف.',
    },
    video: {
        title: 'إضافة فيديو جديد',
        fileLabel: 'ملف الفيديو (MP4, MOV, AVI, WEBM)',
        accept: 'video/mp4,video/quicktime,video/x-msvideo,video/webm',
        urlLabel: 'رابط الصورة المصغرة (اختياري)',
        urlPlaceholder: 'إذا تركته فارغًا، سيتم استخدام أيقونة.',
    },
    audio: {
        title: 'إضافة ملف صوتي جديد',
        fileLabel: 'ملف الصوت (MP3, WAV, OGG, M4A)',
        accept: 'audio/mpeg,audio/wav,audio/ogg,audio/mp4',
        urlLabel: 'رابط صورة الغلاف (اختياري)',
        urlPlaceholder: 'إذا تركته فارغًا، سيتم استخدام أيقونة.',
    },
    image: {
        title: 'إضافة صورة جديدة',
        fileLabel: 'ملف الصورة (JPG, PNG, GIF, WEBP)',
        accept: 'image/jpeg,image/png,image/gif,image/webp',
        urlLabel: 'رابط الصورة (اختياري)',
        urlPlaceholder: 'سيتم استخدام الصورة المرفوعة إذا ترك فارغًا.',
    }
};

const AddMediaModal: React.FC<AddMediaModalProps> = ({ mediaType, onClose, onAdd }) => {
    const [title, setTitle] = useState('');
    const [description, setDescription] = useState('');
    const [url, setUrl] = useState('');
    const [file, setFile] = useState<File | null>(null);
    const [error, setError] = useState('');
    
    // State for video link addition
    const [addMethod, setAddMethod] = useState<'upload' | 'link'>('upload');
    const [youtubeLink, setYoutubeLink] = useState('');

    const currentConfig = config[mediaType];

    const handleSubmit = (e: React.FormEvent) => {
        e.preventDefault();

        if (mediaType === 'video' && addMethod === 'link') {
            if (!youtubeLink.trim()) {
                setError('يرجى إدخال رابط يوتيوب.');
                return;
            }
            // Use a marker file to send the link to the handler
            const markerFile = new File([youtubeLink.trim()], "YOUTUBE_LINK.txt", { type: "text/plain" });
            onAdd(title, description, url, markerFile);
            return;
        }

        if (!file) {
            setError('يرجى رفع ملف.');
            return;
        }
        onAdd(title, description, url, file);
    };
    
    const renderUploadForm = () => (
         <div>
            <label className="block text-sm font-medium text-slate-700">{currentConfig.fileLabel} *</label>
            <div className="mt-1 flex items-center">
                <label className="cursor-pointer bg-white py-2 px-4 border border-slate-300 rounded-md shadow-sm text-sm font-medium text-slate-700 hover:bg-slate-50">
                    <span>اختر ملفاً</span>
                    <input
                        type="file"
                        className="sr-only"
                        onChange={(e) => setFile(e.target.files ? e.target.files[0] : null)}
                        accept={currentConfig.accept}
                        required={addMethod === 'upload'}
                    />
                </label>
                <span className="mr-3 text-sm text-slate-500">{file ? file.name : ''}</span>
            </div>
        </div>
    );

    const renderLinkForm = () => (
         <div>
            <label htmlFor="youtubeLink" className="block text-sm font-medium text-slate-700">رابط يوتيوب *</label>
            <input
                type="url"
                id="youtubeLink"
                value={youtubeLink}
                onChange={(e) => setYoutubeLink(e.target.value)}
                className="mt-1 block w-full px-3 py-2 bg-white border border-slate-300 rounded-md shadow-sm focus:outline-none focus:ring-blue-500 focus:border-blue-500"
                placeholder="https://www.youtube.com/watch?v=..."
                required={addMethod === 'link'}
            />
        </div>
    );


    return (
        <div 
            className="fixed inset-0 bg-black bg-opacity-50 flex justify-center items-center z-50"
            onClick={onClose}
        >
            <div 
                className="bg-white p-8 rounded-2xl shadow-2xl w-full max-w-lg m-4"
                onClick={e => e.stopPropagation()}
            >
                <h2 className="text-2xl font-bold text-slate-800 mb-6">{currentConfig.title}</h2>
                {error && <p className="text-red-500 mb-4">{error}</p>}
                <form onSubmit={handleSubmit} className="space-y-4">
                    
                    {mediaType === 'video' && (
                        <div className="flex border-b border-slate-200 mb-4">
                            <button type="button" onClick={() => setAddMethod('upload')} className={`px-4 py-2 text-sm font-medium ${addMethod === 'upload' ? 'border-b-2 border-blue-500 text-blue-600' : 'text-slate-500 hover:text-slate-700'}`}>رفع ملف</button>
                            <button type="button" onClick={() => setAddMethod('link')} className={`px-4 py-2 text-sm font-medium ${addMethod === 'link' ? 'border-b-2 border-blue-500 text-blue-600' : 'text-slate-500 hover:text-slate-700'}`}>إضافة باستخدام الرابط</button>
                        </div>
                    )}
                    
                    <div>
                        <label htmlFor="title" className="block text-sm font-medium text-slate-700">العنوان (اختياري)</label>
                        <input
                            type="text"
                            id="title"
                            value={title}
                            onChange={(e) => setTitle(e.target.value)}
                            className="mt-1 block w-full px-3 py-2 bg-white border border-slate-300 rounded-md shadow-sm focus:outline-none focus:ring-blue-500 focus:border-blue-500"
                            placeholder="سيتم ملء هذا الفراغ بالذكاء الاصطناعي"
                        />
                    </div>
                    <div>
                        <label htmlFor="description" className="block text-sm font-medium text-slate-700">الوصف (اختياري)</label>
                        <textarea
                            id="description"
                            value={description}
                            onChange={(e) => setDescription(e.target.value)}
                            rows={3}
                            className="mt-1 block w-full px-3 py-2 bg-white border border-slate-300 rounded-md shadow-sm focus:outline-none focus:ring-blue-500 focus:border-blue-500"
                            placeholder="سيتم ملء هذا الفراغ بالذكاء الاصطناعي"
                        />
                    </div>
                    <div>
                        <label htmlFor="imageUrl" className="block text-sm font-medium text-slate-700">{currentConfig.urlLabel}</label>
                        <input
                            type="url"
                            id="imageUrl"
                            value={url}
                            onChange={(e) => setUrl(e.target.value)}
                            className="mt-1 block w-full px-3 py-2 bg-white border border-slate-300 rounded-md shadow-sm focus:outline-none focus:ring-blue-500 focus:border-blue-500"
                            placeholder="سيتم ملء هذا الفراغ بالذكاء الاصطناعي"
                        />
                    </div>
                    
                    {mediaType === 'video' ? (
                        addMethod === 'upload' ? renderUploadForm() : renderLinkForm()
                    ) : (
                        renderUploadForm()
                    )}

                    <div className="flex justify-end gap-4 pt-4">
                        <button
                            type="button"
                            onClick={onClose}
                            className="px-4 py-2 bg-slate-200 text-slate-800 font-semibold rounded-lg hover:bg-slate-300"
                        >
                            إلغاء
                        </button>
                        <button
                            type="submit"
                            className="px-4 py-2 bg-blue-600 text-white font-semibold rounded-lg hover:bg-blue-700"
                        >
                            إضافة
                        </button>
                    </div>
                </form>
            </div>
        </div>
    );
};

export default AddMediaModal;