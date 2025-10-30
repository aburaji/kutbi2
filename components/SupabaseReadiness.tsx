import React from 'react';

const ReadinessCheckItem: React.FC<{ title: string; description: string; }> = ({ title, description }) => (
    <div className="flex items-start gap-4">
        <div className="flex-shrink-0 w-6 h-6 bg-green-100 rounded-full flex items-center justify-center mt-1">
            <svg className="w-4 h-4 text-green-600" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor">
                <path fillRule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clipRule="evenodd" />
            </svg>
        </div>
        <div>
            <h4 className="font-semibold text-slate-800">{title}</h4>
            <p className="text-sm text-slate-600">{description}</p>
        </div>
    </div>
);

const SupabaseReadiness: React.FC = () => {
    return (
        <div className="p-6 bg-white rounded-2xl shadow-lg border border-slate-200">
            <div className="text-center mb-8">
                <h2 className="text-2xl font-bold text-slate-800">هل التطبيق جاهز لرفعه على Supabase؟</h2>
                <p className="mt-2 text-lg font-bold text-green-600">نعم، التطبيق جاهز بالكامل!</p>
            </div>
            
            <div className="space-y-6 max-w-2xl mx-auto">
                <p className="text-slate-700 text-center">
                    تم بناء هذا التطبيق بتقنيات حديثة تجعله متوافقًا تمامًا مع خدمات الاستضافة الثابتة (Static Hosting) مثل Supabase. إليك النقاط التي تؤكد جاهزيته:
                </p>

                <ReadinessCheckItem
                    title="لا يعتمد على خادم"
                    description="كل العمليات المعقدة (مثل تحليل النصوص) تتم في متصفح المستخدم مباشرةً باستخدام Gemini API، مما يجعله مثالياً للاستضافة الثابتة."
                />

                <ReadinessCheckItem
                    title="يعمل بدون خطوة بناء (No Build Step)"
                    description="يتم ترجمة الكود مباشرة في المتصفح، مما يعني أنه يمكنك رفع الملفات كما هي إلى Supabase Hosting دون الحاجة إلى أدوات بناء معقدة."
                />

                <ReadinessCheckItem
                    title="إدارة البيانات من جانب العميل"
                    description="البيانات التي يضيفها المستخدم (مثل الكتب والملاحظات) تُحفظ محليًا في متصفح المستخدم باستخدام IndexedDB. هذا يعني عدم الحاجة إلى قاعدة بيانات على الخادم."
                />
                
                <ReadinessCheckItem
                    title="إدارة مفتاح API من جانب العميل"
                    description="يعتمد التطبيق على مفتاح Gemini API الذي يقدمه المستخدم ويتم تخزينه محليًا في متصفحك فقط. لا توجد حاجة لمتغيرات بيئة على الخادم، مما يجعل النشر على Supabase مباشرًا وآمنًا."
                />
            </div>

            <div className="mt-10 pt-6 border-t border-slate-200 text-center">
                <h3 className="text-lg font-semibold text-slate-800">خطوات النشر المقترحة على Supabase:</h3>
                <ol className="list-decimal list-inside text-left max-w-xl mx-auto mt-4 space-y-2 text-slate-600">
                    <li>اذهب إلى لوحة تحكم مشروعك على Supabase.</li>
                    <li>من القائمة الجانبية، اختر "Hosting".</li>
                    <li>اربط مستودع GitHub الخاص بك الذي يحتوي على ملفات المشروع.</li>
                    <li>اترك "Base directory" فارغًا (أو اضبطه حسب هيكل مستودعك).</li>
                    <li>لا حاجة لـ "Build command" لأن التطبيق لا يتطلب خطوة بناء.</li>
                    <li>انشر الموقع!</li>
                </ol>
            </div>
        </div>
    );
};

export default SupabaseReadiness;