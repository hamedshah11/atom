#!/usr/bin/env python3
"""
Quick Setup Script for O3-mini Configuration
Configures the Business Idea Analyzer to use O3-mini reasoning model
"""

import os
import sys
from pathlib import Path

def check_o3mini_availability():
    """Test if O3-mini is available with current API key"""
    try:
        from openai import OpenAI
        
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            return None, "No API key set"
        
        client = OpenAI(api_key=api_key)
        
        # Test O3-mini
        try:
            response = client.chat.completions.create(
                model="o3-mini",
                messages=[{"role": "user", "content": "test"}],
                max_completion_tokens=10,
                reasoning_effort="low"
            )
            return True, "O3-mini is available!"
        except Exception as e:
            error_msg = str(e)
            if "tier" in error_msg.lower() or "quota" in error_msg.lower():
                # Test GPT-4o fallback
                try:
                    response = client.chat.completions.create(
                        model="gpt-4o",
                        messages=[{"role": "user", "content": "test"}],
                        max_tokens=10
                    )
                    return False, "O3-mini requires Tier 1+ (GPT-4o fallback available)"
                except:
                    return False, "Neither O3-mini nor GPT-4o available"
            else:
                return False, f"O3-mini error: {error_msg[:100]}"
    except ImportError:
        return None, "OpenAI library not installed"
    except Exception as e:
        return None, f"Error: {str(e)[:100]}"

def setup_o3mini():
    """Configure the application for O3-mini"""
    
    print("🧠 Business Idea Analyzer - O3-mini Setup")
    print("=" * 50)
    
    # Check for OpenAI API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("\n⚠️  OpenAI API Key not found")
        api_key = input("Enter your OpenAI API Key: ").strip()
        if api_key:
            os.environ["OPENAI_API_KEY"] = api_key
            print("✅ API Key set for this session")
    else:
        print("✅ OpenAI API Key found")
    
    # Test O3-mini availability
    print("\n🔍 Testing O3-mini availability...")
    available, message = check_o3mini_availability()
    
    if available is None:
        print(f"⚠️  {message}")
    elif available:
        print(f"✅ {message}")
        print("   O3-mini will be used for all analysis")
    else:
        print(f"⚠️  {message}")
        print("   The system will use GPT-4o as fallback")
    
    # Set model configuration
    os.environ["MODEL_ALL"] = "o3-mini"
    print("\n📝 Configuration set to O3-mini")
    
    # Update .env file
    env_path = Path(".env")
    if env_path.exists():
        print("\n📄 Updating .env file...")
        
        with open(env_path, 'r') as f:
            lines = f.readlines()
        
        # Update MODEL_ALL line
        updated = False
        for i, line in enumerate(lines):
            if line.startswith("MODEL_ALL="):
                lines[i] = "MODEL_ALL=o3-mini\n"
                updated = True
                break
        
        if not updated:
            lines.append("\nMODEL_ALL=o3-mini\n")
        
        # Update RATE_LIMIT_DELAY for O3-mini
        delay_updated = False
        for i, line in enumerate(lines):
            if line.startswith("RATE_LIMIT_DELAY="):
                lines[i] = "RATE_LIMIT_DELAY=1.2\n"
                delay_updated = True
                break
        
        if not delay_updated:
            lines.append("RATE_LIMIT_DELAY=1.2\n")
        
        with open(env_path, 'w') as f:
            f.writelines(lines)
        
        print("✅ .env file updated for O3-mini")
    
    # Create Streamlit secrets
    streamlit_dir = Path(".streamlit")
    secrets_path = streamlit_dir / "secrets.toml"
    
    if not streamlit_dir.exists():
        streamlit_dir.mkdir()
    
    print("\n📝 Creating Streamlit secrets...")
    
    secrets_content = f'''# Streamlit Secrets for O3-mini
openai_api_key = "{api_key if api_key else 'YOUR_API_KEY_HERE'}"
model_all = "o3-mini"
verbosity = "low"

# Optional: LangSmith Configuration
# langchain_api_key = "YOUR_LANGSMITH_KEY"
# langchain_tracing_v2 = "true"
# langchain_project = "business-idea-analyzer-o3mini"
'''
    
    with open(secrets_path, 'w') as f:
        f.write(secrets_content)
    
    print("✅ Streamlit secrets configured")
    
    # Summary
    print("\n" + "=" * 50)
    print("✅ O3-mini Setup Complete!")
    print(f"\n📊 Configuration Summary:")
    print(f"   Primary Model: o3-mini")
    print(f"   Fallback Model: gpt-4o")
    print(f"   Rate Limit: 1.2 seconds")
    print(f"   API Key: {'Configured' if (api_key or os.getenv('OPENAI_API_KEY')) else 'Not set'}")
    
    # Important notes
    print("\n⚠️  Important Notes:")
    print("   1. O3-mini requires API Tier 1+ ($5+ spend)")
    print("   2. System auto-falls back to GPT-4o if needed")
    print("   3. O4-mini doesn't exist yet (as of Oct 2025)")
    
    # Instructions
    print("\n🚀 To run the application:")
    print("   streamlit run ui/app.py")
    
    print("\n💡 Tips:")
    print("   - O3-mini provides advanced reasoning")
    print("   - Supports streaming & tool calling (O1 doesn't)")
    print("   - ~$0.15 per analysis (2x GPT-4o cost)")
    print("   - Best for complex business analysis")
    
    return "o3-mini"

def test_analysis():
    """Quick test of the O3-mini setup"""
    print("\n" + "=" * 50)
    print("🧪 Quick Test")
    
    test = input("\nRun a quick test analysis? (y/n): ").strip().lower()
    if test == 'y':
        try:
            # Set debug mode
            os.environ["DEBUG_MODE"] = "1"
            os.environ["MODEL_ALL"] = "o3-mini"
            
            from backend import agents
            
            print("\n🔄 Testing Planner Agent with O3-mini...")
            result = agents.planner_agent("Online marketplace for vintage furniture")
            
            if result:
                print("\n✅ Test successful!")
                print("\nSample output:")
                print(result[:200] + "...")
                
                if "Tier restriction" in result:
                    print("\n⚠️  Note: O3-mini unavailable, used GPT-4o fallback")
            else:
                print("\n❌ No response generated")
                
        except Exception as e:
            print(f"\n❌ Test failed: {str(e)[:200]}")

if __name__ == "__main__":
    try:
        model = setup_o3mini()
        test_analysis()
        print("\n✨ Ready to analyze business ideas with O3-mini reasoning!")
    except KeyboardInterrupt:
        print("\n\n⚠️  Setup cancelled")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Setup error: {e}")
        sys.exit(1)
