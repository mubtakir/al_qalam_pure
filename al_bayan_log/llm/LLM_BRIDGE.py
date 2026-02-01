import os
import sys
import json
from datetime import datetime

# Force UTF-8 for Windows Console
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except:
        pass

# Ensure we can import core components if needed
sys.path.append(os.path.join(os.path.dirname(__file__), "Al_Qalam", "core"))
try:
    from immune_system import ImmuneSystem
except ImportError:
    ImmuneSystem = None

class LLMBridge:
    """
    🌉 The Bridge: Connects Al-Qalam's Logic to a Donor Brain (Qwen/Llama).
    """
    def __init__(self, model_path=None, n_ctx=4096):
        self.model_path = model_path
        self.root_dir = os.path.dirname(os.path.abspath(__file__))
        self.models_dir = os.path.join(self.root_dir, "model", "external")
        self.n_ctx = n_ctx
        self.llm = None
        
        # Initialize Immune System
        memory_dir = os.path.join(self.root_dir, "Al_Qalam", "memory")
        if ImmuneSystem:
            self.immune_system = ImmuneSystem(memory_dir)
        else:
            print("⚠️ Immune System not found. Running in unsafe mode!")
            self.immune_system = None
        
        # Auto-detect GGUF in directory
        
        # Auto-detect GGUF in directory
        if not self.model_path:
            self._find_any_model()
            
        self._ensure_model_exists()
        self._load_model()
        
    def _find_any_model(self):
        """Auto-discover any .gguf file in the external folder."""
        if not os.path.exists(self.models_dir):
            return

        files = [f for f in os.listdir(self.models_dir) if f.endswith(".gguf")]
        if not files:
            # Fallback default
            self.model_name = "qwen2.5-7b-instruct-q4_k_m.gguf"
            self.model_path = os.path.join(self.models_dir, self.model_name)
            return

        # Smart Selection Logic:
        # 1. Prefer single files (no "-of-" in name)
        # 2. If split, MUST pick the first part ("-00001-of-")
        
        # Check for split parts first to ensure we pick entry point
        split_entry_points = [f for f in files if "00001-of-" in f]
        single_files = [f for f in files if "-of-" not in f]
        
        if single_files:
            # Pick largest single file
            self.model_name = max(single_files, key=lambda x: os.path.getsize(os.path.join(self.models_dir, x)))
        elif split_entry_points:
            # Pick largest entry point (likely highest quant)
            self.model_name = max(split_entry_points, key=lambda x: os.path.getsize(os.path.join(self.models_dir, x)))
        else:
            # Fallback: Just pick whatever is largest and hope llama.cpp handles it or user fixes it
            self.model_name = max(files, key=lambda x: os.path.getsize(os.path.join(self.models_dir, x)))
            
        self.model_path = os.path.join(self.models_dir, self.model_name)
        print(f"🔍 Auto-detected Donor Brain: {self.model_name}")
        
    def _ensure_model_exists(self):
        """Check if model exists, else provide instructions/download."""
        if not os.path.exists(self.models_dir):
            os.makedirs(self.models_dir)
            
        if not os.path.exists(self.model_path):
            print(f"⚠️ Model not found at: {self.model_path}")
            print(f"⬇️ Please download '{self.model_name}' and place it in '{self.models_dir}'")
            # In a real scenario, we could auto-download from HF here
            # url = "https://huggingface.co/Qwen/Qwen2.5-7B-Instruct-GGUF/resolve/main/qwen2.5-7b-instruct-q4_k_m.gguf"
            raise FileNotFoundError(f"Donor Brain missing! Expected at {self.model_path}")

    def _load_model(self):
        """Load the GGUF model using available backend (llama-cpp or ctransformers)."""
        if not self.model_path or not os.path.exists(self.model_path):
            print("❌ Model path invalid.")
            return

        print(f"🔌 Connecting to Donor Brain: {os.path.basename(self.model_path)}...")
        
        # Method 1: Try llama-cpp-python (Preferred)
        try:
            from llama_cpp import Llama
            self.backend = "llama_cpp"
            self.llm = Llama(
                model_path=self.model_path,
                n_ctx=self.n_ctx,
                n_gpu_layers=-1,
                verbose=False
            )
            print("✅ Brain Transplant Successful! Logic Online (Backend: llama-cpp).")
            return
        except ImportError:
            pass
        except Exception as e:
            print(f"⚠️ llama-cpp load failed: {e}")

        # Method 2: Try ctransformers (Fallback)
        try:
            from ctransformers import AutoModelForCausalLM
            self.backend = "ctransformers"
            
            try:
                # Try GPU first
                self.llm = AutoModelForCausalLM.from_pretrained(
                    self.model_path, 
                    model_type="qwen",
                    context_length=self.n_ctx,
                    gpu_layers=50
                )
                print("✅ Brain Transplant Successful! Logic Online (Backend: ctransformers-GPU).")
            except OSError as e:
                # DLL error (missing CUDA?), fallback to CPU
                print(f"⚠️ GPU init failed ({e}), switching to CPU...")
                try:
                    self.llm = AutoModelForCausalLM.from_pretrained(
                        self.model_path, 
                        model_type="qwen",
                        context_length=self.n_ctx,
                        gpu_layers=0
                    )
                except Exception as qwen_error:
                    print(f"⚠️ Qwen type failed ({qwen_error}), trying 'llama' type...")
                    self.llm = AutoModelForCausalLM.from_pretrained(
                        self.model_path, 
                        model_type="llama",
                        context_length=self.n_ctx,
                        gpu_layers=0
                    )
                print("✅ Brain Transplant Successful! Logic Online (Backend: ctransformers-CPU).")

            return
        except ImportError:
            pass
        except Exception as e:
            print(f"❌ ctransformers load failed: {e}")

        # Method 3: Try External Binary (Subprocess)
        # Look for llama-cli.exe or main.exe in probable paths
        potential_bins = [
            r"C:\Users\allmy\Desktop\baserah_ai\llama-b7898-bin-win-cpu-x64\llama-cli.exe",
            os.path.join(self.root_dir, "llama_bin", "llama-cli.exe"),
            os.path.join(self.root_dir, "llama_bin", "main.exe"),
            os.path.join(self.root_dir, "llama.cpp", "build", "bin", "Release", "llama-cli.exe"),
            "llama-cli.exe" # PATH
        ]
        
        for bin_path in potential_bins:
            if os.path.exists(bin_path):
                self.backend = "binary"
                self.binary_path = bin_path
                print(f"✅ Brain Transplant Successful! Logic Online (Backend: Binary '{os.path.basename(bin_path)}').")
                return

        print("❌ ALL METHODS FAILED.")
        print("1. Python libraries (llama-cpp-python, ctransformers) failed.")
        print("2. Binary executable (llama-cli.exe) not found.")
        print("➡️ Please download 'llama-bXXXX-bin-win-avx2-x64.zip' from https://github.com/ggerganov/llama.cpp/releases")
        print("   and extract it to a folder named 'llama_bin' in this directory.")

    def think(self, prompt, system_prompt="You are Al-Qalam, a rational AI engine."):
        """Query the Donor Brain."""
        
        # Format for Qwen/Llama 3 (ChatML/Instruct)
        full_prompt = f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
        
        try:
            if getattr(self, 'backend', '') == 'llama_cpp':
                output = self.llm(
                    full_prompt,
                    max_tokens=1024,
                    stop=["<|im_end|>", "</s>"],
                    echo=False,
                    temperature=0.7
                )
                return output['choices'][0]['text'].strip()
                
            elif getattr(self, 'backend', '') == 'ctransformers':
                response = self.llm(
                    full_prompt, 
                    max_new_tokens=1024,
                    stop=["<|im_end|>", "</s>"],
                    temperature=0.7
                )
                return response
                
            elif getattr(self, 'backend', '') == 'binary':
                # Call external executable
                import subprocess
                cmd = [
                    self.binary_path,
                    "-m", self.model_path,
                    "-p", full_prompt,
                    "-n", "1024",
                    "--temp", "0.7",
                    "-c", str(self.n_ctx),
                    "--no-display-prompt",
                    "--log-disable" 
                ]
                
                # Qwen chat template handling might be weird in raw binary without -chat-template override
                # But let's try raw prompt passing.
                
                result = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8', errors='ignore')
                if result.returncode != 0:
                    return f"Binary Error: {result.stderr}"
                
                # Clean up output (remove prompt if echoed)
                output = result.stdout.strip()
                # Basic cleanup if it echoes prompt despite --no-display-prompt issues in some versions
                if full_prompt in output:
                    output = output.replace(full_prompt, "")
                
                return output.strip()
                
        except Exception as e:
            return f"Thinking Error: {e}"

    def think_and_validate(self, prompt, system_prompt="You are an expert Python developer."):
        """
        Think, extract code, and validate it via the Immune System.
        Returns: {"valid": bool, "code": str, "error": str, "raw_response": str}
        """
        raw_response = self.think(prompt, system_prompt)
        
        # Extract code (looking for ```python or just ```)
        import re
        code_match = re.search(r"```python(.*?)```", raw_response, re.DOTALL)
        if not code_match:
            code_match = re.search(r"```(.*?)```", raw_response, re.DOTALL)
            
        if not code_match:
            return {"valid": False, "code": None, "error": "No code block found", "raw_response": raw_response}
            
        code = code_match.group(1).strip()
        
        if self.immune_system:
            validation = self.immune_system.validate_code(code)
            if not validation["valid"]:
                return {"valid": False, "code": code, "error": validation["error"], "raw_response": raw_response}
        
        return {"valid": True, "code": code, "error": None, "raw_response": raw_response}

if __name__ == "__main__":
    # Test the bridge
    print("🌉 Testing LLM Bridge...")
    try:
        bridge = LLMBridge()
        response = bridge.think("مرحبا، من أنت وماذا تستطيع أن تفعل؟")
        print(f"\n🤖 Response:\n{response}")
    except Exception as e:
        print(f"\n⚠️ Test Skipped: {e}")
