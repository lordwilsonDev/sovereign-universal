import SwiftUI

// ============================================================================
// 🏛️ SOVEREIGN DASHBOARD - MAIN APP
// ============================================================================

@main
struct SovereignDashboardApp: App {
    var body: some Scene {
        WindowGroup {
            MainSovereignView()
                .frame(minWidth: 900, minHeight: 700)
                .background(Color.black)
        }
        .windowStyle(.hiddenTitleBar)
    }
}

// ============================================================================
// 🎯 MAIN VIEW
// ============================================================================

struct MainSovereignView: View {
    @StateObject var telemetry = TelemetryStore()
    @StateObject var tribunal = TribunalLogStore()
    @StateObject var axiomHealth = AxiomHealthStore()
    
    var body: some View {
        HStack(spacing: 0) {
            // Sidebar: System Axioms & Health
            AxiomSidebarView(health: axiomHealth)
                .frame(width: 250)
                .background(Color(white: 0.08))
            
            Divider()
                .background(Color.purple.opacity(0.3))
            
            // Main Content: Telemetry & MoIE Active Conversation
            VStack(spacing: 0) {
                TelemetryHeader(metrics: telemetry.currentMetrics)
                
                Divider()
                    .background(Color.purple.opacity(0.3))
                
                TribunalLiveFeed(logs: tribunal.entries)
                
                Divider()
                    .background(Color.purple.opacity(0.3))
                
                ControlPanel(tribunal: tribunal, axiomHealth: axiomHealth)
            }
            .background(Color(white: 0.05))
        }
        .onAppear {
            telemetry.startMonitoring()
            tribunal.startSimulation()
        }
    }
}

// ============================================================================
// 📊 DATA STORES
// ============================================================================

class TelemetryStore: ObservableObject {
    @Published var currentMetrics = TelemetryMetrics()
    private var timer: Timer?
    
    func startMonitoring() {
        timer = Timer.scheduledTimer(withTimeInterval: 2.0, repeats: true) { [weak self] _ in
            self?.updateMetrics()
        }
        updateMetrics()
    }
    
    private func updateMetrics() {
        // Simulate real metrics (in production, read from telemetry.py JSON)
        DispatchQueue.main.async {
            self.currentMetrics = TelemetryMetrics(
                cpuUsage: Double.random(in: 30...60),
                memoryUsedGB: Double.random(in: 6...9),
                memoryTotalGB: 16.0,
                thermalState: ["nominal", "fair"].randomElement()!,
                diskReadMB: Double.random(in: 0...5),
                diskWriteMB: Double.random(in: 0...3)
            )
        }
    }
}

struct TelemetryMetrics {
    var cpuUsage: Double = 0
    var memoryUsedGB: Double = 0
    var memoryTotalGB: Double = 16.0
    var thermalState: String = "nominal"
    var diskReadMB: Double = 0
    var diskWriteMB: Double = 0
    
    var memoryAvailableGB: Double { memoryTotalGB - memoryUsedGB }
    var memoryPercent: Double { memoryUsedGB / memoryTotalGB }
    var isHot: Bool { thermalState == "serious" || thermalState == "critical" }
    var abundanceViolation: Bool { memoryAvailableGB < 2 || isHot || cpuUsage > 85 }
}

class TribunalLogStore: ObservableObject {
    @Published var entries: [TribunalEntry] = []
    private var timer: Timer?
    
    func startSimulation() {
        // Add initial entries
        addEntry(.architect, "Initializing Sovereign Stack...")
        addEntry(.system, "MoIE Tribunal ready")
        addEntry(.critic, "Inversion Critic online. Scanning for gaps...")
    }
    
    func addEntry(_ role: TribunalRole, _ message: String) {
        let entry = TribunalEntry(role: role, message: message, timestamp: Date())
        DispatchQueue.main.async {
            self.entries.append(entry)
            if self.entries.count > 100 {
                self.entries.removeFirst()
            }
        }
    }
    
    func simulateFailure() {
        addEntry(.system, "⚠️ EMERGENCY OVERRIDE TRIGGERED")
        addEntry(.system, "🔥 Simulating Black Swan Event...")
        
        DispatchQueue.main.asyncAfter(deadline: .now() + 1) {
            self.addEntry(.architect, "CRITICAL: axiom_module.py corrupted!")
            
            DispatchQueue.main.asyncAfter(deadline: .now() + 1) {
                self.addEntry(.executioner, "Phoenix Kernel activated. Initiating resurrection...")
                
                DispatchQueue.main.asyncAfter(deadline: .now() + 2) {
                    self.addEntry(.critic, "Scanning chaos_tests.py for required signatures...")
                    
                    DispatchQueue.main.asyncAfter(deadline: .now() + 2) {
                        self.addEntry(.architect, "Re-deriving logic from Core Axioms...")
                        
                        DispatchQueue.main.asyncAfter(deadline: .now() + 2) {
                            self.addEntry(.executioner, "Building new module (116 lines)...")
                            
                            DispatchQueue.main.asyncAfter(deadline: .now() + 1) {
                                self.addEntry(.system, "✅ RESURRECTION COMPLETE - Sovereignty Restored")
                            }
                        }
                    }
                }
            }
        }
    }
    
    func clearLogs() {
        entries.removeAll()
        addEntry(.system, "Logs cleared")
    }
}

enum TribunalRole: String {
    case architect = "ARCHITECT"
    case executioner = "EXECUTIONER"
    case critic = "CRITIC"
    case system = "SYSTEM"
    
    var color: Color {
        switch self {
        case .architect: return .blue
        case .executioner: return .orange
        case .critic: return .purple
        case .system: return .green
        }
    }
    
    var icon: String {
        switch self {
        case .architect: return "building.columns"
        case .executioner: return "hammer"
        case .critic: return "eye"
        case .system: return "gearshape"
        }
    }
}

struct TribunalEntry: Identifiable {
    let id = UUID()
    let role: TribunalRole
    let message: String
    let timestamp: Date
    
    var timeString: String {
        let formatter = DateFormatter()
        formatter.dateFormat = "HH:mm:ss"
        return formatter.string(from: timestamp)
    }
}

class AxiomHealthStore: ObservableObject {
    @Published var love: AxiomState = .healthy
    @Published var safety: AxiomState = .healthy
    @Published var abundance: AxiomState = .healthy
    @Published var growth: AxiomState = .healthy
    @Published var resurrectionCount: Int = 0
    
    func triggerAnomaly(_ axiom: Axiom) {
        switch axiom {
        case .love: love = .warning
        case .safety: safety = .critical
        case .abundance: abundance = .warning
        case .growth: growth = .warning
        }
        
        // Auto-recover after 5 seconds
        DispatchQueue.main.asyncAfter(deadline: .now() + 5) {
            self.resetAll()
        }
    }
    
    func incrementResurrection() {
        resurrectionCount += 1
    }
    
    func resetAll() {
        love = .healthy
        safety = .healthy
        abundance = .healthy
        growth = .healthy
    }
}

enum Axiom: String, CaseIterable {
    case love = "Love"
    case safety = "Safety"
    case abundance = "Abundance"
    case growth = "Growth"
}

enum AxiomState {
    case healthy
    case warning
    case critical
    
    var color: Color {
        switch self {
        case .healthy: return .green
        case .warning: return .yellow
        case .critical: return .red
        }
    }
}

// ============================================================================
// 🛡️ AXIOM SIDEBAR
// ============================================================================

struct AxiomSidebarView: View {
    @ObservedObject var health: AxiomHealthStore
    
    var body: some View {
        VStack(alignment: .leading, spacing: 20) {
            // Header
            HStack {
                Image(systemName: "shield.checkmark.fill")
                    .foregroundColor(.purple)
                Text("AXIOM HEALTH")
                    .font(.headline)
                    .foregroundColor(.white)
            }
            .padding(.bottom, 10)
            
            // Four Axioms
            AxiomIndicator(
                name: "Love",
                symbol: "λ",
                icon: "heart.fill",
                color: .red,
                state: health.love,
                description: "Cohesion Check"
            )
            
            AxiomIndicator(
                name: "Safety",
                symbol: "σ",
                icon: "shield.fill",
                color: .blue,
                state: health.safety,
                description: "CBF Status"
            )
            
            AxiomIndicator(
                name: "Abundance",
                symbol: "α",
                icon: "sparkles",
                color: .yellow,
                state: health.abundance,
                description: "Efficiency Check"
            )
            
            AxiomIndicator(
                name: "Growth",
                symbol: "γ",
                icon: "leaf.fill",
                color: .green,
                state: health.growth,
                description: "Evolution Status"
            )
            
            Divider()
                .background(Color.gray.opacity(0.3))
            
            // Resurrection Counter
            VStack(alignment: .leading, spacing: 8) {
                Text("RESURRECTION COUNT")
                    .font(.caption)
                    .foregroundColor(.gray)
                
                HStack {
                    Image(systemName: "arrow.counterclockwise")
                        .foregroundColor(.purple)
                    Text("\(health.resurrectionCount)")
                        .font(.title)
                        .fontWeight(.bold)
                        .foregroundColor(.white)
                }
            }
            
            Spacer()
            
            // Version
            Text("Sovereign Stack v1.0")
                .font(.caption2)
                .foregroundColor(.gray)
        }
        .padding()
    }
}

struct AxiomIndicator: View {
    let name: String
    let symbol: String
    let icon: String
    let color: Color
    let state: AxiomState
    let description: String
    
    @State private var isFlashing = false
    
    var body: some View {
        HStack(spacing: 12) {
            // Icon with state color
            ZStack {
                Circle()
                    .fill(state.color.opacity(0.2))
                    .frame(width: 40, height: 40)
                
                Image(systemName: icon)
                    .foregroundColor(state == .healthy ? color : state.color)
                    .opacity(isFlashing ? 0.3 : 1.0)
            }
            
            VStack(alignment: .leading, spacing: 2) {
                HStack {
                    Text(symbol)
                        .font(.caption)
                        .foregroundColor(.gray)
                    Text(name)
                        .font(.subheadline)
                        .fontWeight(.medium)
                        .foregroundColor(.white)
                }
                Text(description)
                    .font(.caption2)
                    .foregroundColor(.gray)
            }
            
            Spacer()
            
            // Status dot
            Circle()
                .fill(state.color)
                .frame(width: 8, height: 8)
        }
        .padding(.vertical, 8)
        .padding(.horizontal, 12)
        .background(Color.white.opacity(0.05))
        .cornerRadius(8)
        .onChange(of: state) { newState in
            if newState != .healthy {
                withAnimation(.easeInOut(duration: 0.5).repeatForever()) {
                    isFlashing = true
                }
            } else {
                isFlashing = false
            }
        }
    }
}

// ============================================================================
// 📊 TELEMETRY HEADER
// ============================================================================

struct TelemetryHeader: View {
    let metrics: TelemetryMetrics
    
    var body: some View {
        HStack(spacing: 30) {
            // CPU Gauge
            GaugeView(
                value: metrics.cpuUsage / 100,
                label: "CPU",
                valueText: String(format: "%.0f%%", metrics.cpuUsage),
                color: metrics.cpuUsage > 85 ? .red : .blue
            )
            
            // Memory Gauge
            GaugeView(
                value: metrics.memoryPercent,
                label: "Memory",
                valueText: String(format: "%.1f GB", metrics.memoryAvailableGB),
                color: metrics.memoryAvailableGB < 2 ? .red : .green
            )
            
            // Thermal Status
            VStack(spacing: 8) {
                Image(systemName: thermalIcon)
                    .font(.title)
                    .foregroundColor(thermalColor)
                
                Text(metrics.thermalState.uppercased())
                    .font(.caption)
                    .foregroundColor(.white)
                
                Text("Thermal")
                    .font(.caption2)
                    .foregroundColor(.gray)
            }
            .frame(width: 80)
            
            Spacer()
            
            // Status Badge
            HStack {
                Circle()
                    .fill(metrics.abundanceViolation ? Color.orange : Color.green)
                    .frame(width: 8, height: 8)
                
                Text(metrics.abundanceViolation ? "INVERSION REQUIRED" : "SOVEREIGN")
                    .font(.caption)
                    .foregroundColor(metrics.abundanceViolation ? .orange : .green)
            }
            .padding(.horizontal, 16)
            .padding(.vertical, 8)
            .background(Color.white.opacity(0.05))
            .cornerRadius(20)
        }
        .padding()
        .background(Color(white: 0.08))
    }
    
    var thermalIcon: String {
        switch metrics.thermalState {
        case "nominal": return "snowflake"
        case "fair": return "thermometer.medium"
        case "serious": return "flame"
        case "critical": return "flame.fill"
        default: return "thermometer"
        }
    }
    
    var thermalColor: Color {
        switch metrics.thermalState {
        case "nominal": return .blue
        case "fair": return .yellow
        case "serious": return .orange
        case "critical": return .red
        default: return .gray
        }
    }
}

struct GaugeView: View {
    let value: Double // 0-1
    let label: String
    let valueText: String
    let color: Color
    
    var body: some View {
        VStack(spacing: 8) {
            ZStack {
                Circle()
                    .stroke(Color.gray.opacity(0.2), lineWidth: 8)
                    .frame(width: 60, height: 60)
                
                Circle()
                    .trim(from: 0, to: value)
                    .stroke(color, style: StrokeStyle(lineWidth: 8, lineCap: .round))
                    .frame(width: 60, height: 60)
                    .rotationEffect(.degrees(-90))
                
                Text(valueText)
                    .font(.caption2)
                    .fontWeight(.bold)
                    .foregroundColor(.white)
            }
            
            Text(label)
                .font(.caption2)
                .foregroundColor(.gray)
        }
    }
}

// ============================================================================
// 📜 TRIBUNAL LIVE FEED
// ============================================================================

struct TribunalLiveFeed: View {
    let logs: [TribunalEntry]
    
    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            // Header
            HStack {
                Image(systemName: "terminal")
                    .foregroundColor(.green)
                Text("MoIE ACTIVE CONVERSATION")
                    .font(.caption)
                    .foregroundColor(.gray)
                Spacer()
                Text("\(logs.count) entries")
                    .font(.caption2)
                    .foregroundColor(.gray)
            }
            .padding()
            .background(Color(white: 0.08))
            
            // Log entries
            ScrollViewReader { proxy in
                ScrollView {
                    LazyVStack(alignment: .leading, spacing: 4) {
                        ForEach(logs) { entry in
                            LogEntryView(entry: entry)
                                .id(entry.id)
                        }
                    }
                    .padding()
                }
                .onChange(of: logs.count) { _ in
                    if let lastEntry = logs.last {
                        withAnimation {
                            proxy.scrollTo(lastEntry.id, anchor: .bottom)
                        }
                    }
                }
            }
            .background(Color.black)
            .font(.system(.caption, design: .monospaced))
        }
    }
}

struct LogEntryView: View {
    let entry: TribunalEntry
    
    var body: some View {
        HStack(alignment: .top, spacing: 8) {
            Text("[\(entry.timeString)]")
                .foregroundColor(.gray)
            
            Image(systemName: entry.role.icon)
                .foregroundColor(entry.role.color)
                .frame(width: 16)
            
            Text(entry.role.rawValue)
                .foregroundColor(entry.role.color)
                .fontWeight(.medium)
            
            Text(entry.message)
                .foregroundColor(.white)
        }
    }
}

// ============================================================================
// 🎮 CONTROL PANEL
// ============================================================================

struct ControlPanel: View {
    @ObservedObject var tribunal: TribunalLogStore
    @ObservedObject var axiomHealth: AxiomHealthStore
    
    var body: some View {
        HStack(spacing: 16) {
            // Emergency Override Button
            Button(action: {
                axiomHealth.triggerAnomaly(.safety)
                axiomHealth.incrementResurrection()
                tribunal.simulateFailure()
            }) {
                HStack {
                    Image(systemName: "flame.fill")
                    Text("TRIGGER BLACK SWAN")
                }
            }
            .buttonStyle(.borderedProminent)
            .tint(.red)
            
            // Clear Logs
            Button(action: {
                tribunal.clearLogs()
            }) {
                HStack {
                    Image(systemName: "trash")
                    Text("Clear Logs")
                }
            }
            .buttonStyle(.bordered)
            
            Spacer()
            
            // System Status
            HStack(spacing: 8) {
                Circle()
                    .fill(Color.green)
                    .frame(width: 8, height: 8)
                Text("System: SOVEREIGN")
                    .font(.caption)
                    .foregroundColor(.green)
            }
            
            // Black Swan Labs Badge
            Text("🦢 Black Swan Labs")
                .font(.caption)
                .foregroundColor(.purple)
        }
        .padding()
        .background(Color(white: 0.08))
    }
}

// ============================================================================
// 🎨 PREVIEW
// ============================================================================

struct MainSovereignView_Previews: PreviewProvider {
    static var previews: some View {
        MainSovereignView()
            .frame(width: 900, height: 700)
    }
}
