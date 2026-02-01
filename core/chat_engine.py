import re
import random
from typing import Dict, List, Optional, Tuple
from core.linguistic_narrator import LinguisticNarrator

# Arabic NLP from Bayan
try:
    from al_bayan_log.nlp import segment, pos_tag, extract_entities
    ARABIC_NLP_AVAILABLE = True
except ImportError:
    ARABIC_NLP_AVAILABLE = False

class ChatEngine:
    """Parses natural language into commands for the SelfWritingModel."""
    
    def __init__(self, model):
        self.model = model
        self.narrator = LinguisticNarrator()
        self.last_inference: Optional[List[str]] = None
        self.last_subject: Optional[str] = None
        
        # Extended Arabic Patterns with Synonyms
        self.patterns = {
            "intro": [r"مرحبا", r"أهلا", r"سلام", r"يا قلم", r"من أنت", r"عرفني بنفسك"],
            "learn": [
                r"تعلم أن (.*) هم (.*)",
                r"أضف مفهوم (.*) وهم (.*)",
                r"تعلم أن (.*) هي (.*)",
                r"من هم (.*)\? هم (.*)"
            ],
            "fact": [
                r"(.*) (يصمم|يحب|يعمل في|يسكن في|يقطن في|يعيش في|يهوى|يبني|يشتغل) (.*)",
                r"أضف حقيقة أن (.*) (.*) (.*)"
            ],
            "pronoun_fact": [
                r"^(هو|هي) (يصمم|يحب|يعمل في|يسكن في|يقطن في|يعيش في|يهوى|يبني|يشتغل) (.*)"
            ],
            "query": [
                r"ماذا تعرف عن (.*)\?",
                r"من هو (.*)\?",
                r"أخبرني عن (.*)",
                r"حدثني عن (.*)"
            ],
            "feedback_pos": [r"صحيح", r"نعم", r"مضبوط", r"أيوة", r"بالضبط", r"تماما", r"موافق"],
            "feedback_neg": [r"خطأ", r"لا", r"غير صحيح", r"مرفوض", r"كلا", r"ليس كذلك"],
            "induce": [r"استنتج", r"حلل البيانات", r"دمج المنطق", r"فكر قليلا", r"ماذا استنتجت"]
        }

    def process(self, text: str) -> str:
        text = text.strip()
        
        # 0. Intro Handler
        for p in self.patterns.get("intro", []):
            if re.search(p, text):
                return random.choice(self.narrator.templates["intro"])

        # 1. Feedback handling (requires context)
        if self.last_inference:
            if any(re.match(p, text) for p in self.patterns["feedback_pos"]):
                for rule in self.last_inference:
                    self.model.apply_feedback(rule, positive=True)
                self.last_inference = None
                return "ممتاز، قمت بتطوير الأوزان البرمجية بناءً على تأكيدك."
            
            if any(re.match(p, text) for p in self.patterns["feedback_neg"]):
                for rule in self.last_inference:
                    self.model.apply_feedback(rule, positive=False)
                self.last_inference = None
                return "فهمت، قمت بتعديل الأوزان لتقليل ثقتي في هذا الاستنتاج."

        # 2. Learn Concept
        for p in self.patterns["learn"]:
            match = re.search(p, text)
            if match:
                name = match.group(1).strip()
                examples = [ex.strip() for ex in match.group(2).split("و")]
                self.model.learn_concept(name, examples)
                self.last_subject = name
                return f"تم تعلم مفهوم '{name}' بنجاح مع الأمثلة: {', '.join(examples)}."

        # 2.5 Pronoun Fact
        for p in self.patterns.get("pronoun_fact", []):
            match = re.search(p, text)
            if match and self.last_subject:
                pred, obj = match.group(2).strip(), match.group(3).strip()
                self.model.add_fact(self.last_subject, pred, obj)
                return f"فهمت، تقصد {self.last_subject}. تم تسجيل أن {self.last_subject} {pred} {obj}."

        # 3. Add Fact
        for p in self.patterns["fact"]:
            match = re.search(p, text)
            if match:
                subj, pred, obj = match.group(1).strip(), match.group(2).strip(), match.group(3).strip()
                self.last_subject = subj
                self.model.add_fact(subj, pred, obj)
                return f"تم تسجيل حقيقة: {subj} {pred} {obj}."

        # 4. Induce Logic
        if any(re.search(p, text) for p in self.patterns["induce"]):
            # Capture rules before they are added to avoid re-triggering? 
            # SelfWritingModel already handles duplicates.
            self.model.induce_rules()
            return "انتهيت من تحليل الأنماط وتوليد قواعد منطقية جديدة."

        # 5. Query
        for p in self.patterns["query"]:
            match = re.search(p, text)
            if match:
                query_val = match.group(1).strip()
                cell = self.model._find_cell_by_meta_value(query_val)
                if cell:
                    self.last_subject = query_val
                    return self.model.query_knowledge(query_val)
                return f"عذراً، لا أجد في ذاكرتي ما يخص '{query_val}' حالياً."

        # 6. Fallback - Try Inference
        triggered = self.model.infer({})
        if triggered:
            self.last_inference = triggered
            # Narrate the first inference for beauty
            eloquent = self.narrator.narrate_inference(triggered[0])
            return f"{eloquent}\nهل هذا ما كنت ترمي إليه؟"

        return self.narrator.templates["intro"][random.randint(0,2)] if "مرحبا" in text else "لم أفهم طلبك تماماً، هل يمكنك صياغته بشكل أوضح؟"

    def analyze_text(self, text: str) -> Dict:
        """Analyzes Arabic text using Bayan NLP for deeper understanding."""
        if not ARABIC_NLP_AVAILABLE:
            return {"error": "Arabic NLP not available"}
        
        return {
            "segments": segment(text),
            "pos_tags": pos_tag(text),
            "entities": extract_entities(text)
        }
