

import React, { useState, useEffect, useCallback } from 'react';
import { Book, Video } from '../types.ts';

interface CarouselProps {
    items: (Book | Video)[];
    onItemClick: (item: Book | Video) => void;
}

const Carousel: React.FC<CarouselProps> = ({ items, onItemClick }) => {
    const [currentIndex, setCurrentIndex] = useState(0);

    const goToPrevious = useCallback(() => {
        if (items.length === 0) return;
        const isFirstSlide = currentIndex === 0;
        const newIndex = isFirstSlide ? items.length - 1 : currentIndex - 1;
        setCurrentIndex(newIndex);
    }, [currentIndex, items.length]);

    const goToNext = useCallback(() => {
        if (items.length === 0) return;
        const isLastSlide = currentIndex === items.length - 1;
        const newIndex = isLastSlide ? 0 : currentIndex + 1;
        setCurrentIndex(newIndex);
    }, [currentIndex, items.length]);

    useEffect(() => {
        if (items.length > 1) {
            const slideInterval = setInterval(goToNext, 5000); // Auto-slide every 5 seconds
            return () => clearInterval(slideInterval);
        }
    }, [goToNext, items.length]);

    if (!items || items.length === 0) {
        return null;
    }
    
    const currentItem = items[currentIndex];
    // This check prevents an error if items array is valid but currentIndex is out of bounds somehow.
    if (!currentItem) {
        return null;
    }

    const isBook = 'imageUrl' in currentItem;
    const imageUrl = isBook ? (currentItem as Book).imageUrl : (currentItem as Video).thumbnailUrl;

    const [imgSrc, setImgSrc] = useState(imageUrl);

    useEffect(() => {
        setImgSrc(imageUrl); // Reset image source when the current item changes
    }, [imageUrl]);


    return (
        <div className="relative w-full aspect-[16/7] rounded-2xl shadow-lg border border-slate-200 overflow-hidden mb-6 bg-slate-800 group">
            <div className="absolute inset-0 w-full h-full bg-black bg-opacity-40 z-10"></div>
            <img 
                src={imgSrc} 
                alt={currentItem.title} 
                className="absolute inset-0 w-full h-full object-cover transition-transform duration-500 group-hover:scale-105" 
                onError={() => setImgSrc('https://images.unsplash.com/photo-1589998059171-988d887df646?q=80&w=1200&auto=format&fit=crop')}
            />
            
            <div className="relative z-20 h-full flex flex-col justify-end p-6 md:p-8 text-white">
                <span className={`text-xs font-bold px-3 py-1 rounded-full mb-2 inline-block self-start ${isBook ? 'bg-blue-500' : 'bg-purple-500'}`}>
                    {isBook ? 'كتاب مميز' : 'فيديو مميز'}
                </span>
                <h2 className="text-2xl md:text-3xl font-bold mb-2 shadow-text">{currentItem.title}</h2>
                <p className="text-sm md:text-base line-clamp-2 max-w-2xl shadow-text mb-4">{currentItem.description}</p>
                <button 
                    onClick={() => onItemClick(currentItem)}
                    className="bg-white text-slate-800 font-semibold py-2 px-5 rounded-lg hover:bg-slate-200 transition-colors duration-300 self-start shadow-md"
                >
                    ابدأ التحليل الآن
                </button>
            </div>
            
            {/* Navigation Buttons */}
            {items.length > 1 && (
                <>
                    <button onClick={goToPrevious} className="absolute top-1/2 -translate-y-1/2 left-4 z-30 bg-black bg-opacity-30 text-white rounded-full p-2 hover:bg-opacity-50 transition-opacity opacity-0 group-hover:opacity-100" aria-label="السابق">
                        {/* FIX: Changed numeric strokeWidth property to a string for consistency. */}
                        <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M15 19l-7-7 7-7" /></svg>
                    </button>
                    <button onClick={goToNext} className="absolute top-1/2 -translate-y-1/2 right-4 z-30 bg-black bg-opacity-30 text-white rounded-full p-2 hover:bg-opacity-50 transition-opacity opacity-0 group-hover:opacity-100" aria-label="التالي">
                        {/* FIX: Changed numeric strokeWidth property to a string for consistency. */}
                        <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M9 5l7 7-7 7" /></svg>
                    </button>

                    {/* Dots */}
                    <div className="absolute bottom-4 left-1/2 -translate-x-1/2 z-30 flex space-x-2">
                        {items.map((_, index) => (
                            <button 
                                key={index} 
                                onClick={() => setCurrentIndex(index)}
                                className={`w-2 h-2 rounded-full transition-colors ${currentIndex === index ? 'bg-white' : 'bg-white bg-opacity-40 hover:bg-opacity-70'}`}
                                aria-label={`الانتقال إلى الشريحة ${index + 1}`}
                            ></button>
                        ))}
                    </div>
                </>
            )}
        </div>
    );
};

export default Carousel;