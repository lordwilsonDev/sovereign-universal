import json

def tool_deployer(request):
    """
    Vertex AI Cloud Function: Tool_Deployer.
    Automates the deployment of child agent containers to Cloud Run.
    """
    try:
        data = request.get_json()
        plan = data.get("plan", {})
        
        # Simulation of container build and push
        agent_name = plan.get("agent_name", "untitled-agent")
        print(f"Building Docker image for {agent_name}...")
        
        deployment_res = {
            "agent": agent_name,
            "status": "DEPLOYED",
            "url": f"https://{agent_name.lower().replace('_', '-')}.a.run.app",
            "timestamp": "2026-03-16"
        }

        return json.dumps({"status": "SUCCESS", "deployment": deployment_res}), 200, {"Content-Type": "application/json"}
    except Exception as e:
        return json.dumps({"status": "ERROR", "reason": str(e)}), 500, {"Content-Type": "application/json"}
 Lands on Cloud Run in us-west1.
