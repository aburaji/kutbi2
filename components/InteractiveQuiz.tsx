import React, { useState } from 'react';
import { QuizQuestion } from '../types.ts';

interface InteractiveQuizProps {
    questions: QuizQuestion[];
    onComplete: (answers: (number | null)[]) => void;
    bookTitle: string;
}

const InteractiveQuiz: React.FC<InteractiveQuizProps> = ({ questions, onComplete, bookTitle }) => {
    const [currentQuestionIndex, setCurrentQuestionIndex] = useState(0);
    const [userAnswers, setUserAnswers] = useState<(number | null)[]>(new Array(questions.length).fill(null));
    const [selectedOption, setSelectedOption] = useState<number | null>(null);

    const handleOptionSelect = (optionIndex: number) => {
        setSelectedOption(optionIndex);
    };

    const handleNextQuestion = () => {
        // Save the current answer
        const newAnswers = [...userAnswers];
        newAnswers[currentQuestionIndex] = selectedOption;
        setUserAnswers(newAnswers);

        // Reset selected option for the next question
        setSelectedOption(null);

        // Move to the next question or complete the quiz
        if (currentQuestionIndex < questions.length - 1) {
            setCurrentQuestionIndex(currentQuestionIndex + 1);
        } else {
            onComplete(newAnswers);
        }
    };

    const currentQuestion = questions[currentQuestionIndex];
    const progressPercentage = ((currentQuestionIndex + 1) / questions.length) * 100;

    return (
        <div className="p-6 bg-white rounded-2xl shadow-lg border border-slate-200">
            <h2 className="text-xl font-bold text-slate-800 mb-2">اختبار قصير: {bookTitle}</h2>
            <p className="text-slate-600 mb-4">السؤال {currentQuestionIndex + 1} من {questions.length}</p>
            
            {/* Progress Bar */}
            <div className="w-full bg-slate-200 rounded-full h-2.5 mb-6">
                <div className="bg-blue-600 h-2.5 rounded-full" style={{ width: `${progressPercentage}%` }}></div>
            </div>

            <div className="mb-6">
                <h3 className="text-lg font-semibold text-slate-800">{currentQuestion.question}</h3>
            </div>

            <div className="space-y-3">
                {currentQuestion.options.map((option, index) => (
                    <button
                        key={index}
                        onClick={() => handleOptionSelect(index)}
                        className={`w-full text-right p-4 rounded-lg border-2 transition-all duration-200
                            ${selectedOption === index 
                                ? 'bg-blue-100 border-blue-500 ring-2 ring-blue-300' 
                                : 'bg-slate-50 border-slate-200 hover:bg-slate-100 hover:border-slate-300'}`}
                    >
                        {option}
                    </button>
                ))}
            </div>

            <div className="mt-8 flex justify-end">
                <button
                    onClick={handleNextQuestion}
                    disabled={selectedOption === null}
                    className="px-6 py-2 bg-blue-600 text-white font-semibold rounded-lg hover:bg-blue-700 disabled:bg-slate-400 disabled:cursor-not-allowed"
                >
                    {currentQuestionIndex < questions.length - 1 ? 'التالي' : 'إنهاء الاختبار'}
                </button>
            </div>
        </div>
    );
};

export default InteractiveQuiz;