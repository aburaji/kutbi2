import React from 'react';

interface DeleteConfirmationModalProps {
    bookTitle: string;
    onConfirm: () => void;
    onCancel: () => void;
}

const DeleteConfirmationModal: React.FC<DeleteConfirmationModalProps> = ({ bookTitle, onConfirm, onCancel }) => {
    return (
        <div 
            className="fixed inset-0 bg-black bg-opacity-60 flex justify-center items-center z-50"
            onClick={onCancel}
        >
            <div 
                className="bg-white p-8 rounded-2xl shadow-2xl w-full max-w-md m-4 text-center"
                onClick={e => e.stopPropagation()}
            >
                <h2 className="text-2xl font-bold text-slate-800 mb-4">تأكيد الحذف</h2>
                <p className="text-slate-600 mb-8">
                    هل أنت متأكد أنك تريد حذف كتاب <span className="font-bold">"{bookTitle}"</span>؟
                    <br/>
                    لا يمكن التراجع عن هذا الإجراء.
                </p>
                <div className="flex justify-center gap-4">
                    <button
                        onClick={onCancel}
                        className="px-6 py-2 bg-slate-200 text-slate-800 font-semibold rounded-lg hover:bg-slate-300"
                    >
                        إلغاء
                    </button>
                    <button
                        onClick={onConfirm}
                        className="px-6 py-2 bg-red-600 text-white font-semibold rounded-lg hover:bg-red-700"
                    >
                        تأكيد الحذف
                    </button>
                </div>
            </div>
        </div>
    );
};

export default DeleteConfirmationModal;