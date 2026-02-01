import random
from typing import Dict, List, Any

class LinguisticNarrator:
    """Transforms raw logical inferences into eloquent Arabic narration."""
    
    def __init__(self):
        # Templates for different relations and types of inferences
        self.templates = {
            "induction": [
                "بناءً على ملاحظاتي لـ {concept}، يبدو أن هناك ميلاً طبيعياً لديهم نحو {target}.",
                "من خلال تحليل الأنماط، وجدت أن {concept} غالباً ما يرتبطون بـ {target}.",
                "لقد استنتجت حكمة جديدة: إن طبيعة {concept} تدفعهم دائماً نحو {target}.",
                "ثمة رابط وثيق اكتشفته؛ فحيثما وُجد {concept}، وُجد الاهتمام بـ {target}."
            ],
            "fact": [
                "لقد سُجل في ذاكرتي أن {subject} {relation} {object}.",
                "من الحقائق المؤكدة لدي أن {subject} {relation} {object}.",
                "أعلم يقيناً أن {subject} {relation} {object}."
            ],
            "analogy": [
                "قياساً على {name1} وارتباطه بـ {target}، يجول في خاطري أن {name2} قد يشترك معه في هذا الأمر.",
                "بما أن {name1} يميل لـ {target}، فليس من المستبعد أن يكون لـ {name2} شأن مشابه.",
                "هل تظن أن {name2} يقتفي أثر {name1} في صلته بـ {target}؟"
            ],
            "dream": [
                "في لحظة تأمل وافتراض، تساءلت: ماذا لو كان {concept} {relation} {target}؟",
                "راودني حدس في حالة الحلم يصور لي {concept} وهو {relation} {target}."
            ],
            "intro": [
                "أهلاً بك في رحاب المنطق الحي.",
                "أنا القلم، أصيغ لك من البيانات بياناً.",
                "مرحباً بك، تزداد بصيرتي بحديثك."
            ]
        }

    def narrate_thought(self, thought: str) -> str:
        """Attempts to rephrase a technical thought into something more natural."""
        # Simple phrase detection for now
        if "قياس منطقي" in thought:
            # Rephrase analogy: Includes pattern matching or splitting
            return self._eloquent_analogy(thought)
        
        if "ملاحظة: مفهوم" in thought:
             return thought.replace("ملاحظة: مفهوم", "لقد لاحظت أن مفهوم").replace("لديه بيانات أقل", "ما زال يحتاج لمزيد من الإيضاح والبيانات")
        
        if "حدس من الأحلام" in thought:
             return self._eloquent_dream(thought)

        return thought

    def _eloquent_analogy(self, thought: str) -> str:
        # Expected format: "قياس منطقي: بما أن {name1} يرتبط بـ {target}، هل تفكر أن {name2} قد يرتبط به أيضاً؟"
        try:
            temp = thought.split("بما أن ")[1]
            name1 = temp.split(" يرتبط بـ ")[0]
            target = temp.split(" يرتبط بـ ")[1].split("،")[0]
            name2 = thought.split("هل تفكر أن ")[1].split(" قد يرتبط")[0]
            
            template = random.choice(self.templates["analogy"])
            return template.format(name1=name1, name2=name2, target=target)
        except:
            return thought

    def _eloquent_dream(self, thought: str) -> str:
        # Expected format: "حدس من الأحلام: يبدو منطقياً أن {concept} {relation} {target}. هل نعتمد هذا؟"
        try:
            core = thought.split("يبدو منطقياً أن ")[1].split(". هل نعتمد")[0]
            parts = core.split(" ")
            concept = parts[0]
            relation = parts[1]
            target = " ".join(parts[2:])
            
            template = random.choice(self.templates["dream"])
            return template.format(concept=concept, relation=relation, target=target) + " هل توافقني هذا التصور؟"
        except:
            return thought

    def narrate_inference(self, rule_name: str) -> str:
        """Translates a rule name into a descriptive sentence."""
        # rule_induct_concept_engineer_item_bracket
        parts = rule_name.split("_")
        if "induct" in parts:
            try:
                concept_idx = parts.index("concept")
                item_idx = parts.index("item")
                concept = parts[concept_idx + 1]
                target = parts[item_idx + 1]
                
                # Manual map for some concepts if needed, or use formatted strings
                template = random.choice(self.templates["induction"])
                return template.format(concept=concept, target=target)
            except:
                return f"لقد استخلصت قاعدة جديدة تتعلق بـ {rule_name}."
        
        return f"تم تطبيق القاعدة: {rule_name}."
