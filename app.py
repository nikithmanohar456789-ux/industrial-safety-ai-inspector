import argparse
import base64
import json
import os
import tempfile
from io import BytesIO
from typing import Any, Dict, Optional

import requests
from PIL import Image
from langchain_groq import ChatGroq
from crewai import Agent, Crew, LLM, Process, Task
from crewai.tools import tool
from dotenv import load_dotenv
from crewai import LLM

load_dotenv()

# ============================================================
# Config
# ============================================================
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
# TEXT_MODEL = os.getenv("TEXT_MODEL", "ollama/qwen3:4b")
TEXT_MODEL = os.getenv("TEXT_MODEL", "groq/llama-3.1-8b-instant")
VISION_MODEL = os.getenv("VISION_MODEL", "qwen3-vl:4b")

DEFAULT_USER_QUERY = (
    "Inspect this image for industrial safety risks, PPE compliance, visible hazards, "
    "machinery risk, warning signs, and anything needing supervisor attention."
)

DEFAULT_POLICY = """
Safety rules:
1. Every worker in an active industrial area should wear a safety helmet.
2. Gloves are required near machinery and sharp tools unless the task explicitly forbids them.
3. Restricted zones must not contain unauthorized personnel.
4. Visible warning signage must be respected.
5. Forklift and pedestrian zones should be clearly separated.
6. Sparks, smoke, exposed wiring, spills, or blocked exits are high-priority risks.
"""


# ============================================================
# Helpers
# ============================================================
def ensure_file_exists(path: str) -> None:
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")


def read_text_file(path: str) -> str:
    ensure_file_exists(path)
    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip()


def image_path_to_base64(image_path: str, max_size: int = 1400) -> str:
    ensure_file_exists(image_path)

    image = Image.open(image_path).convert("RGB")
    image.thumbnail((max_size, max_size))

    buffer = BytesIO()
    image.save(buffer, format="JPEG", quality=90)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def parse_json_safely(text: str) -> Dict[str, Any]:
    text = (text or "").strip()

    try:
        return json.loads(text)
    except Exception:
        pass

    if "```json" in text:
        text = text.split("```json", 1)[1]
        text = text.split("```", 1)[0].strip()
    elif "```" in text:
        text = text.split("```", 1)[1]
        text = text.split("```", 1)[0].strip()

    try:
        return json.loads(text)
    except Exception:
        return {
            "raw_output": text,
            "warning": "Model did not return valid JSON."
        }


def pretty_json(obj: Any) -> str:
    return json.dumps(obj, indent=2, ensure_ascii=False)


# ============================================================
# Direct Ollama Calls
# ============================================================
def call_ollama_generate(prompt: str, model: str, image_path: Optional[str] = None) -> str:
    payload: Dict[str, Any] = {
        "model": model.replace("ollama/", ""),
        "prompt": prompt,
        "stream": False,
    }

    if image_path:
        payload["images"] = [image_path_to_base64(image_path)]

    response = requests.post(
        f"{OLLAMA_BASE_URL}/api/generate",
        json=payload,
        timeout=240,
    )
    response.raise_for_status()
    data = response.json()
    text = data.get("response", "")

    if not text or not text.strip():
        raise ValueError(f"Empty response from model: {model}")

    return text.strip()


def repair_json_if_needed(raw_text: str, fallback_schema_hint: str = "") -> Dict[str, Any]:
    parsed = parse_json_safely(raw_text)
    if "warning" not in parsed:
        return parsed

    repair_prompt = f"""
Convert the following output into valid JSON only.
Do not add markdown.
Do not add explanation.

Schema hint:
{fallback_schema_hint}

Original output:
{raw_text}
"""
    repaired = call_ollama_generate(prompt=repair_prompt, model=TEXT_MODEL)
    return parse_json_safely(repaired)


# ============================================================
# Manual Vision Steps
# ============================================================

def run_scene_analysis(image_path: str) -> Dict[str, Any]:
    prompt = """
You are an industrial vision inspection assistant.

Analyze the image and return ONLY valid JSON.
Do not add markdown.
Do not add explanation text.

Schema:
{
  "scene_type": "indoor/outdoor/warehouse/factory/construction/other",
  "people_count": 0,
  "people": [
    {
      "id": "worker_1",
      "role_guess": "worker/operator/visitor/unknown",
      "helmet": "yes/no/unclear",
      "gloves": "yes/no/unclear",
      "vest": "yes/no/unclear",
      "near_machine": "yes/no/unclear"
    }
  ],
  "machinery": ["forklift", "conveyor", "drill", "unknown"],
  "zones": ["pedestrian zone", "restricted area", "loading area"],
  "hazards": ["sparks", "spill", "blocked exit", "exposed wiring"],
  "signage_visible": true,
  "summary": "short grounded summary",
  "uncertainties": ["..."]
}

Rules:
- Be conservative.
- If unclear, say "unclear".
- Do not invent details.
- Return JSON only.
"""
    raw = call_ollama_generate(prompt=prompt, model=VISION_MODEL, image_path=image_path)
    return repair_json_if_needed(
        raw,
        fallback_schema_hint="""
{
  "scene_type": "...",
  "people_count": 0,
  "people": [],
  "machinery": [],
  "zones": [],
  "hazards": [],
  "signage_visible": false,
  "summary": "...",
  "uncertainties": []
}
""",
    )


# ======================================================================
# creating scene_analysis_tool for CrewAI
# ======================================================================
@tool("Scene Analysis Tool")
def scene_analysis_tool(image_path: str) -> str:
    """
    Analyze an industrial image and return structured scene understanding.

    This tool MUST be used when you need:
    - scene type (factory, warehouse, etc.)
    - people and PPE compliance
    - machinery presence
    - hazards and zones
    - overall scene summary

    Input:
    - image_path: local file path to the image

    Output:
    - Strict JSON string with keys:
      scene_type, people_count, people, machinery, zones,
      hazards, signage_visible, summary, uncertainties
    """

    try:
        # Call your existing vision function (UNCHANGED logic)
        result = run_scene_analysis(image_path)

        # Ensure valid JSON string output
        return json.dumps(result, ensure_ascii=False)

    except Exception as e:
        # Fail-safe JSON (VERY IMPORTANT for agent stability)
        error_output = {
            "scene_type": "unknown",
            "people_count": 0,
            "people": [],
            "machinery": [],
            "zones": [],
            "hazards": [],
            "signage_visible": False,
            "summary": "Scene analysis failed",
            "uncertainties": [str(e)]
        }

        return json.dumps(error_output, ensure_ascii=False)


def run_ocr_signage(image_path: str) -> Dict[str, Any]:
    prompt = """
You are extracting visible text from an industrial image.

Return VALID JSON ONLY.

Schema:
{
  "visible_text": [
    {
      "text": "exact text if visible",
      "type": "warning_sign/label/panel/screen/other",
      "confidence": "high/medium/low"
    }
  ],
  "notes": "brief note if text is blurry or uncertain"
}

Rules:
- Do not guess hidden text.
- Preserve wording when possible.
- Return JSON only.
"""
    raw = call_ollama_generate(prompt=prompt, model=VISION_MODEL, image_path=image_path)
    return repair_json_if_needed(
        raw,
        fallback_schema_hint="""
{
  "visible_text": [],
  "notes": ""
}
""",
    )

# ============================================================
# Creating ocr_signage_tool for CrewAI
# ============================================================
@tool("OCR Signage Tool")
def ocr_tool(image_path: str) -> str:
    """
    Extract visible text and signage from an industrial image.

    Use this tool when you need:
    - warning signs
    - labels
    - control panel text
    - any readable content in the scene

    Input:
    - image_path: local file path

    Output:
    - JSON string with:
      visible_text, notes
    """

    try:
        result = run_ocr_signage(image_path)
        return json.dumps(result, ensure_ascii=False)

    except Exception as e:
        return json.dumps({
            "visible_text": [],
            "notes": f"OCR failed: {str(e)}"
        }, ensure_ascii=False)


def run_focused_risk_reasoning(image_path: str, user_query: str) -> Dict[str, Any]:
    prompt = f"""
You are a targeted industrial vision assistant.

Focus instruction:
{user_query}

If the instruction is broad, prioritize:
- PPE violations
- sparks
- spills
- blocked exits
- forklift and pedestrian conflict
- machine proximity risk
- restricted zone breach
- warning sign non-compliance

Return VALID JSON ONLY.

Schema:
{{
  "focus_instruction": "{user_query}",
  "findings": ["..."],
  "risk_level": "low/medium/high/unclear",
  "evidence": ["evidence1", "evidence2"],
  "uncertainty": "what remains uncertain"
}}

Rules:
- Stay grounded in the image.
- If uncertain, say uncertain.
- Return JSON only.
"""
    raw = call_ollama_generate(prompt=prompt, model=VISION_MODEL, image_path=image_path)
    return repair_json_if_needed(
        raw,
        fallback_schema_hint="""
{
  "focus_instruction": "...",
  "findings": [],
  "risk_level": "unclear",
  "evidence": [],
  "uncertainty": ""
}
""",
    )

#================================================================
# creating focused_risk_tool for CrewAI
#================================================================   
@tool("Risk Assessment Tool")
def risk_tool(input: str) -> str:
    """
    Analyze safety risks in an industrial image.

    This tool evaluates:
    - PPE violations
    - machinery risks
    - unsafe conditions
    - hazard presence

    Input (JSON string):
    {
        "image_path": "path/to/image",
        "user_query": "inspection focus"
    }

    Output:
    JSON string with:
    - focus_instruction
    - findings
    - risk_level
    - evidence
    - uncertainty
    """     
    try:
        # 👇 FIRST: safely parse input
        try:
            data = json.loads(input)
        except Exception:
            return json.dumps({
                "error": "Invalid JSON input",
                "expected_format": {
                    "image_path": "string",
                    "user_query": "string"
                }
            })

        # 👇 THEN extract values
        image_path = data.get("image_path")
        user_query = data.get("user_query", "")

        result = run_focused_risk_reasoning(image_path, user_query)
        return json.dumps(result, ensure_ascii=False)

    except Exception as e:
        return json.dumps({
            "focus_instruction": "",
            "findings": [],
            "risk_level": "unclear",
            "evidence": [],
            "uncertainty": f"Risk analysis failed: {str(e)}"
        }, ensure_ascii=False)


# ============================================================
# CrewAI only for text reasoning
# ============================================================


def build_llm(model_name: str, temperature: float = 0.1):
    model_name = model_name.replace("groq/", "")

    return LLM(
        model=f"groq/{model_name}",
        api_key=os.getenv("GROQ_API_KEY"),
        temperature=temperature
    )



def build_reasoning_crew(
    user_query: str,
    policy_text: str,
    image_path: str,
) -> Crew:
    shared_text_llm = build_llm(TEXT_MODEL, temperature=0.1)

    scene_analyst = Agent(
    role="Scene Analyst",
    goal="Analyze industrial scenes and extract structured observations",
    backstory=(
        "You are an expert in industrial visual inspection. "
        "You always use the Scene Analysis Tool to extract grounded information. "
        "You never guess or hallucinate."
    ),
    tools=[scene_analysis_tool],
    llm=build_llm(TEXT_MODEL),
    verbose=True,
    allow_delegation=False,
    )
    ocr_agent = Agent(
    role="OCR Specialist",
    goal="Extract visible text and signage from industrial images",
    backstory=(
        "You specialize in reading industrial signs, labels, and warnings. "
        "You always use the OCR Signage Tool to extract text. "
        "Do not guess unreadable text."
    ),
    tools=[ocr_tool],
    llm=shared_text_llm,
    verbose=True,
    allow_delegation=False,
    )


    risk_agent = Agent(
    role="Risk Assessment Expert",
    goal="Identify safety hazards and assess risk levels",
    backstory=(
        "You are an industrial safety expert. "
        "You always use the Risk Assessment Tool to analyze hazards. "
        "You focus on PPE violations, machinery risk, and unsafe conditions."
    ),
    tools=[risk_tool],
    llm=shared_text_llm,
    verbose=True,
    allow_delegation=False,
    )
    
    safety_auditor = Agent(
        role="Safety Compliance Auditor",
        goal="Map visual findings to safety rules and decide violation severity.",
        backstory=(
            "You are a strict but grounded industrial safety auditor. "
            "You must compare observations against policy and avoid unsupported claims."
        ),
        llm=shared_text_llm,
        verbose=True,
        allow_delegation=False,
    )

    incident_reporter = Agent(
        role="Incident Report Writer",
        goal="Produce a clear final report for a plant manager or safety lead.",
        backstory=(
            "You write concise, high-signal incident summaries with executive clarity. "
            "You separate confirmed findings from uncertain findings."
        ),
        llm=shared_text_llm,
        verbose=True,
        allow_delegation=False,
    )

    scene_task = Task(
        description=f"""
Analyze the industrial image located at: {image_path}

You MUST use the Scene Analysis Tool to extract structured information.

Return ONLY valid JSON with:
- scene_type
- people_count
- people
- machinery
- zones
- hazards
- signage_visible
- summary
- uncertainties

Do not guess. Do not skip tool usage.
""",
    agent=scene_analyst,
    expected_output="""
JSON with:
- scene_type
- people_count
- people
- machinery
- zones
- hazards
- signage_visible
- summary
- uncertainties
"""
)
    
    ocr_task = Task(
    description=f"""
Extract all visible text from the image at: {image_path}

You MUST use the OCR Signage Tool.

Return ONLY valid JSON with:
- visible_text
- notes

Do not guess unreadable text.
""",
    agent=ocr_agent,
    expected_output="""
JSON with:
- visible_text
- notes
"""
)
    
    risk_task = Task(
    description=f"""
Analyze safety risks in the image at: {image_path}

Focus instruction:
{user_query}

You MUST call the Risk Assessment Tool with EXACT JSON input:

{{
  "image_path": "{image_path}",
  "user_query": "{user_query}"
}}

Return ONLY valid JSON.
""",
    agent=risk_agent,
    expected_output="""
JSON with:
- focus_instruction
- findings
- risk_level
- evidence
- uncertainty
"""
)  

    audit_task = Task(
        description=f"""
User inspection request:
{user_query}

Safety policy:
{policy_text}


Use outputs from scene analysis, OCR, and risk assessment tasks provided in context.

Return VALID JSON ONLY.

Schema:
{{
  "confirmed_violations": ["..."],
  "possible_violations": ["..."],
  "severity": "low/medium/high",
  "why": ["..."],
  "requires_human_review": ["..."],
  "recommended_immediate_actions": ["..."]
}}

Rules:
- Do not invent facts.
- A violation MUST NOT appear in both confirmed_violations and possible_violations.
- If evidence is uncertain, it MUST ONLY go into possible_violations.
- confirmed_violations should include ONLY high-confidence observations.
- Keep reasoning grounded in provided outputs only.
- Every schema key must be present, even if the value is an empty list.
""",
        expected_output="A JSON audit report with violations, severity, rationale, human review items, and recommended actions.",
        agent=safety_auditor,
        context=[scene_task, ocr_task, risk_task],
    )

    report_task = Task(
        description=f"""
User inspection request:
{user_query}

Use outputs from previous tasks (scene analysis, OCR, and risk assessment)

Use the completed audit task output from context as the authoritative compliance judgment.

Create the final incident report.

Return VALID JSON ONLY.

Schema:
{{
  "executive_summary": "2-4 sentence summary",
  "scene_summary": "grounded scene description",
  "key_findings": ["..."],
  "visible_signage": ["..."],
  "risk_severity": "low/medium/high",
  "recommended_actions": ["..."],
  "uncertainty_note": "what remains uncertain"
}}

Rules:
- Keep it manager-friendly.
- Separate confirmed findings from uncertain ones.
- Prefer the audit task output for violation severity and recommended actions.
- Do not invent facts not present in prior outputs.
- Every schema key must be present.
- risk_severity MUST exactly match the severity from the audit task. 
- Do not override severity.
""",
        expected_output="A final JSON incident report for a manager.",
        agent=incident_reporter,
        context=[scene_task, ocr_task, risk_task, audit_task],
    )

    return Crew(
    agents=[
        scene_analyst,
        ocr_agent,
        risk_agent,
        safety_auditor,
        incident_reporter,
    ],
    tasks=[
        scene_task,
        ocr_task,
        risk_task,
        audit_task,
        report_task,
    ],
    process=Process.sequential,  # keep simple for now
    verbose=True,
)


# ============================================================
# Pipeline
# ============================================================
def run_pipeline(
    image_path: str,
    user_query: Optional[str] = None,
    policy_text: Optional[str] = None,
) -> Dict[str, Any]:
    ensure_file_exists(image_path)

    final_user_query = (user_query or DEFAULT_USER_QUERY).strip()
    final_policy = (policy_text or DEFAULT_POLICY).strip()

    crew = build_reasoning_crew(
        user_query=final_user_query,
        policy_text=final_policy,
        image_path=image_path,
    )

    result = crew.kickoff()

    final_text = str(result)
    final_json = repair_json_if_needed(
        final_text,
        fallback_schema_hint="""
{
  "executive_summary": "...",
  "scene_summary": "...",
  "key_findings": [],
  "visible_signage": [],
  "risk_severity": "low/medium/high",
  "recommended_actions": [],
  "uncertainty_note": ""
}
""",
    )

    return {
        "image_path": image_path,
        "user_query": final_user_query,
        "policy_text": final_policy,
        "final_output_text": final_text,
        "final_output_json": final_json,
    }



# ============================================================
# CLI
# ============================================================
def run_cli(args: argparse.Namespace) -> None:
    policy_text = DEFAULT_POLICY
    if args.policy_file:
        policy_text = read_text_file(args.policy_file)

    output = run_pipeline(
        image_path=args.image,
        user_query=args.query,
        policy_text=policy_text,
    )


    if args.save_json:
        with open(args.save_json, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        print(f"\nSaved full output to: {args.save_json}")


# ============================================================
# Gradio
# ============================================================
def launch_gradio() -> None:
    import gradio as gr

    def gradio_run(image, query, policy):
        if image is None:
            return "Please upload an image.", "{}", "{}", "{}"

        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            temp_path = tmp.name

        try:
            pil_img = Image.fromarray(image).convert("RGB")
            pil_img.save(temp_path, format="JPEG", quality=95)

            output = run_pipeline(
                image_path=temp_path,
                user_query=query or DEFAULT_USER_QUERY,
                policy_text=policy or DEFAULT_POLICY,
            )

            final_json = pretty_json(output["final_output_json"])
            full_json = pretty_json(output)
            summary = output["final_output_json"].get("executive_summary", "No executive_summary found.")

            report_md = f"""# Safety Inspection Report

## Executive Summary
{output["final_output_json"].get("executive_summary", "")}

## Scene Summary
{output["final_output_json"].get("scene_summary", "")}

## Key Findings
{chr(10).join(f"- {x}" for x in output["final_output_json"].get("key_findings", []))}

## Visible Signage
{chr(10).join(f"- {x}" for x in output["final_output_json"].get("visible_signage", []))}

## Risk Severity
{output["final_output_json"].get("risk_severity", "")}

## Recommended Actions
{chr(10).join(f"- {x}" for x in output["final_output_json"].get("recommended_actions", []))}

## Uncertainty Note
{output["final_output_json"].get("uncertainty_note", "")}
"""

            return summary, final_json, full_json, report_md
        except Exception as exc:
            err = f"Error: {exc}"
            return err, "{}", "{}", ""
        finally:
            try:
                os.remove(temp_path)
            except Exception:
                pass

    demo = gr.Interface(
        fn=gradio_run,
        inputs=[
            gr.Image(type="numpy", label="Upload inspection image"),
            gr.Textbox(label="Inspection query", value=DEFAULT_USER_QUERY, lines=3),
            gr.Textbox(label="Policy text", value=DEFAULT_POLICY, lines=10),
        ],
        outputs=[
            gr.Textbox(label="Executive summary"),
            gr.Code(label="Final JSON", language="json"),
            gr.Code(label="Full pipeline output", language="json"),
            gr.Markdown(label="Manager report"),
        ],
        title="CrewAI + Ollama Safety Inspector",
        description="Manual vision stage + CrewAI reasoning stage",
    )

    demo.launch(server_name="127.0.0.1", server_port=7860, share=False)


# ============================================================
# Main
# ============================================================
def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="CrewAI + Ollama industrial safety inspector")
    parser.add_argument("--mode", choices=["cli", "gradio"], default="cli")
    parser.add_argument("--image", type=str, help="Path to input image (required for CLI)")
    parser.add_argument("--query", type=str, default=DEFAULT_USER_QUERY)
    parser.add_argument("--policy-file", type=str, default=None)
    parser.add_argument("--save-json", type=str, default=None)
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    if args.mode == "gradio":
        launch_gradio()
        return

    if not args.image:
        raise ValueError("In CLI mode, --image is required.")

    run_cli(args)


if __name__ == "__main__":
    main()