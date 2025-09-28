from sentence_transformers import SentenceTransformer, util
import torch


class IntentClassifier:
    """Class to classify user intentions using SBERT embeddings"""
    
    def __init__(self, model_name='all-MiniLM-L12-v2', margin_threshold=0.1):
        """
        Initialize the intent classifier
        
        Args:
            model_name (str): Name of the SentenceTransformer model to use
            margin_threshold (float): Minimum difference threshold between top two categories
        """
        self.model_name = model_name
        self.margin_threshold = margin_threshold
        self.model = None
        self.category_embeddings = {}
        
        # Demand templates by category
        self.demand_templates = {
            "read_text": [
            "Can you read aloud what's written here?",
            "Please read the text shown on the screen.",
            "What does the writing say on that sign?",
            "Tell me exactly what the label says.",
            "Could you read the message displayed in front of me?",
            "What are the words written on this surface?",
            "Read this text for me, please."
            ],
            "describe_scene": [
                "Can you describe what's happening around me?",
                "What do you see in this area?",
                "Give me a detailed description of the scene.",
                "Describe the setting and objects nearby.",
                "Tell me what the surroundings look like.",
                "What's visible in the current environment?",
                "What is on my left",
                "What is on my right",
                "What does this room look like?"
            ],
            "activate_detection_collision": [
                "Please enable obstacle and hazard detection.",
                "Turn on the collision prevention system now.",
                "Start the object and movement detection feature.",
                "Activate sensors to detect anything I might bump into.",
                "Can you switch on the obstacle warning system?",
                "Enable collision alerts and monitoring, please."
            ],
            "locate_object": [
                "Where is the phone?",
                "What is to my right?",
                "Where is the person?",
                "Where are the people",
                "What is to my left",
                "Can you find my keys?",
                "Help me locate my wallet?",
                "Where did I leave my glasses?",
                "Can you track down my backpack?"
            ],
            "other": [
                "Play some background music.",
                "What's the weather forecast for today?",
                "Remind me about my 5 PM meeting.",
                "Call my mother's phone.",
                "Open the phone's camera app.",
                "Show directions to the nearest grocery store."
            ]
        }
        
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize the model and compute category embeddings"""
        try:
            print(f"🤖 Loading model {self.model_name}...")
            self.model = SentenceTransformer(self.model_name)
            self._compute_category_embeddings()
            print("✅ Model initialized successfully")
        except Exception as e:
            print(f"❌ Error during model initialization: {e}")
            raise
    
    def _compute_category_embeddings(self):
        """Compute average embeddings for each intent category"""
        try:
            self.category_embeddings = {
                intent: torch.mean(self.model.encode(phrases, convert_to_tensor=True), dim=0)
                for intent, phrases in self.demand_templates.items()
            }
            print(f"📊 Embeddings computed for {len(self.category_embeddings)} categories")
        except Exception as e:
            print(f"❌ Error during embeddings computation: {e}")
            raise
    
    def add_intent_category(self, intent_name, example_phrases):
        """
        Add a new intent category
        
        Args:
            intent_name (str): Name of the new category
            example_phrases (list): List of example phrases for this category
        """
        try:
            if not example_phrases:
                raise ValueError("Example phrases list cannot be empty")
            
            self.demand_templates[intent_name] = example_phrases
            
            # Recalculate embedding for this category
            self.category_embeddings[intent_name] = torch.mean(
                self.model.encode(example_phrases, convert_to_tensor=True), dim=0
            )
            
            print(f"✅ Category '{intent_name}' added with {len(example_phrases)} examples")
            
        except Exception as e:
            print(f"❌ Error adding category '{intent_name}': {e}")
            raise
    
    def remove_intent_category(self, intent_name):
        """
        Remove an intent category
        
        Args:
            intent_name (str): Name of the category to remove
        """
        try:
            if intent_name not in self.demand_templates:
                print(f"⚠️ Category '{intent_name}' does not exist")
                return False
            
            del self.demand_templates[intent_name]
            del self.category_embeddings[intent_name]
            
            print(f"✅ Category '{intent_name}' removed")
            return True
            
        except Exception as e:
            print(f"❌ Error removing category '{intent_name}': {e}")
            return False
    
    def classify(self, text):
        """
        Classify input text into an intent category
        If the top two categories are too close, classify as 'other'
        
        Args:
            text (str): The input sentence to classify
        
        Returns:
            tuple[str, float]: Predicted intent or 'other', and its confidence score
        """
        try:
            if not text or not text.strip():
                return "other", 0.0
            
            if not self.model or not self.category_embeddings:
                raise RuntimeError("Model is not initialized")
            
            # Encode input text
            user_embedding = self.model.encode(text, convert_to_tensor=True)
            
            # Calculate similarities with each category
            similarities = {
                intent: util.cos_sim(user_embedding, emb).item()
                for intent, emb in self.category_embeddings.items()
            }
            
            # Sort by similarity score in descending order
            sorted_sims = sorted(similarities.items(), key=lambda x: x[1], reverse=True)
            
            if not sorted_sims:
                return "other", 0.0
            
            top_intent, top_score = sorted_sims[0]
            second_score = sorted_sims[1][1] if len(sorted_sims) > 1 else 0.0
            
            # Check if difference is sufficient and minimum score is met
            if top_score - second_score <= self.margin_threshold or top_score < 0.2:
                return "other", top_score - second_score
            
            return top_intent, top_score
            
        except Exception as e:
            print(f"❌ Error during classification: {e}")
            return "other", 0.0
    
    def get_intent_categories(self):
        """Return the list of available intent categories"""
        return list(self.demand_templates.keys())
    
    def get_category_examples(self, intent_name):
        """
        Return examples for a given category
        
        Args:
            intent_name (str): Name of the category
            
        Returns:
            list: List of examples for this category
        """
        return self.demand_templates.get(intent_name, [])
    
    def set_margin_threshold(self, threshold):
        """
        Modify the margin threshold for classification
        
        Args:
            threshold (float): New margin threshold
        """
        if threshold < 0:
            raise ValueError("Margin threshold must be positive")
        
        self.margin_threshold = threshold
        print(f"✅ Margin threshold updated: {threshold}")
    
    def get_classification_details(self, text):
        """
        Return complete details about text classification
        
        Args:
            text (str): The text to analyze
            
        Returns:
            dict: Classification details with all scores
        """
        try:
            if not text or not text.strip():
                return {"error": "Empty text"}
            
            if not self.model or not self.category_embeddings:
                return {"error": "Model not initialized"}
            
            # Encode input text
            user_embedding = self.model.encode(text, convert_to_tensor=True)
            
            # Calculate similarities with each category
            similarities = {
                intent: util.cos_sim(user_embedding, emb).item()
                for intent, emb in self.category_embeddings.items()
            }
            
            # Sort by score
            sorted_sims = sorted(similarities.items(), key=lambda x: x[1], reverse=True)
            
            predicted_intent, confidence = self.classify(text)
            
            return {
                "input_text": text,
                "predicted_intent": predicted_intent,
                "confidence": confidence,
                "all_scores": dict(sorted_sims),
                "margin_threshold": self.margin_threshold,
                "model_name": self.model_name
            }
            
        except Exception as e:
            return {"error": f"Error during analysis: {e}"}
    
    def cleanup(self):
        """Clean up model resources"""
        try:
            if self.model:
                self.model = None
                self.category_embeddings = {}
                print("✅ Classifier resources cleaned up")
        except Exception as e:
            print(f"❌ Error during cleanup: {e}")