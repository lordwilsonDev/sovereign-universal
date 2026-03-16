document.addEventListener('DOMContentLoaded', () => {
    const canvas = document.getElementById('chart-canvas');
    const ctx = canvas.getContext('2d');
    const eventFeed = document.getElementById('event-feed');
    const tokenFlux = document.getElementById('token-flux');
    const clock = document.getElementById('system-clock');

    function resize() {
        canvas.width = canvas.parentElement.clientWidth;
        canvas.height = canvas.parentElement.clientHeight - 100; // Account for headers
    }
    resize();
    window.addEventListener('resize', resize);

    // Business Data Visualization (Smooth Curves)
    const points = Array.from({length: 20}, () => Math.random() * 200 + 100);
    
    function drawChart() {
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        
        ctx.strokeStyle = 'rgba(197, 160, 89, 0.5)';
        ctx.lineWidth = 2;
        ctx.beginPath();
        
        const spacing = canvas.width / (points.length - 1);
        
        ctx.moveTo(0, canvas.height - points[0]);
        for(let i=1; i<points.length; i++) {
            const x = i * spacing;
            const y = canvas.height - points[i];
            ctx.lineTo(x, y);
        }
        ctx.stroke();

        // Add glow area
        ctx.lineTo(canvas.width, canvas.height);
        ctx.lineTo(0, canvas.height);
        const gradient = ctx.createLinearGradient(0, 0, 0, canvas.height);
        gradient.addColorStop(0, 'rgba(197, 160, 89, 0.2)');
        gradient.addColorStop(1, 'transparent');
        ctx.fillStyle = gradient;
        ctx.fill();

        // Shift points for animation
        points.shift();
        points.push(Math.random() * 200 + 100);

        requestAnimationFrame(drawChart);
    }
    drawChart();

    // Log & Data Simulation
    const msgs = [
        "HUNYUAN_MODEL_SYNC: OPTIMAL",
        "SOCIAL_REACH_EXPANDED: +14.2%",
        "INTENT_ENGINE: CALIBRATED",
        "SOVEREIGN_LEDGER_UPDATE: VERIFIED",
        "MARKET_SENTIMENT_SCAN: BULLISH"
    ];

    setInterval(() => {
        const line = document.createElement('div');
        line.innerText = `> ${msgs[Math.floor(Math.random() * msgs.length)]}`;
        eventFeed.prepend(line);
        if(eventFeed.children.length > 8) eventFeed.lastChild.remove();

        // Update token flux
        const flux = (Math.random() * 5000 + 10000).toLocaleString();
        tokenFlux.innerText = `${flux} / sec`;

        // Update clock
        const now = new Date();
        clock.innerText = `${now.getHours().toString().padStart(2, '0')}:${now.getMinutes().toString().padStart(2, '0')}:${now.getSeconds().toString().padStart(2, '0')} UTC`;
    }, 2500);

    console.log("📈 Business Command Center Online. High-fidelity enterprise orchestration stabilized.");
});
