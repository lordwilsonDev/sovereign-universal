import SwiftUI
import Foundation

// ============================================================================
// 📊 SYSTEM METRICS (Matching Python telemetry.py)
// ============================================================================

struct SystemMetrics: Codable {
    var cpu_usage: Double
    var memory_used_gb: Double
    var memory_available_gb: Double
    var memory_pressure: String
    var thermal_state: String
    var disk_read_mb: Double
    var disk_write_mb: Double
    var timestamp: String
    var abundance_violation: Bool
}

// ============================================================================
// 🌡️ TELEMETRY SNIFFER (Swift Native)
// ============================================================================

class TelemetrySniffer: ObservableObject {
    @Published var metrics: SystemMetrics = SystemMetrics(
        cpu_usage: 0,
        memory_used_gb: 0,
        memory_available_gb: 16,
        memory_pressure: "normal",
        thermal_state: "nominal",
        disk_read_mb: 0,
        disk_write_mb: 0,
        timestamp: ISO8601DateFormatter().string(from: Date()),
        abundance_violation: false
    )
    
    private var timer: Timer?
    
    func startMonitoring(interval: TimeInterval = 2.0) {
        timer = Timer.scheduledTimer(withTimeInterval: interval, repeats: true) { [weak self] _ in
            self?.sniff()
        }
        sniff() // Immediate first read
    }
    
    func stopMonitoring() {
        timer?.invalidate()
        timer = nil
    }
    
    func sniff() {
        let thermal = getThermalState()
        let memory = getMemoryUsage()
        let cpu = getCPUUsage()
        
        let abundanceViolation = thermal != "nominal" || 
                                  memory.available < 2.0 || 
                                  cpu > 85
        
        DispatchQueue.main.async {
            self.metrics = SystemMetrics(
                cpu_usage: cpu,
                memory_used_gb: memory.used,
                memory_available_gb: memory.available,
                memory_pressure: self.calculatePressure(available: memory.available),
                thermal_state: thermal,
                disk_read_mb: 0,
                disk_write_mb: 0,
                timestamp: ISO8601DateFormatter().string(from: Date()),
                abundance_violation: abundanceViolation
            )
        }
    }
    
    private func getThermalState() -> String {
        switch ProcessInfo.processInfo.thermalState {
        case .nominal: return "nominal"
        case .fair: return "fair"
        case .serious: return "serious"
        case .critical: return "critical"
        @unknown default: return "unknown"
        }
    }
    
    private func getMemoryUsage() -> (used: Double, available: Double) {
        var stats = host_basic_info()
        var count = mach_msg_type_number_t(MemoryLayout<host_basic_info>.size) / 4
        
        let result = withUnsafeMutablePointer(to: &stats) {
            $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
                host_info(mach_host_self(), HOST_BASIC_INFO, $0, &count)
            }
        }
        
        guard result == KERN_SUCCESS else {
            return (0, 16)
        }
        
        let totalGB = Double(stats.max_mem) / (1024 * 1024 * 1024)
        
        // Get VM stats for used memory
        var vmStats = vm_statistics64()
        var vmCount = mach_msg_type_number_t(MemoryLayout<vm_statistics64>.size / MemoryLayout<integer_t>.size)
        
        let vmResult = withUnsafeMutablePointer(to: &vmStats) {
            $0.withMemoryRebound(to: integer_t.self, capacity: Int(vmCount)) {
                host_statistics64(mach_host_self(), HOST_VM_INFO64, $0, &vmCount)
            }
        }
        
        guard vmResult == KERN_SUCCESS else {
            return (totalGB / 2, totalGB / 2)
        }
        
        let pageSize = UInt64(vm_kernel_page_size)
        let active = Double(vmStats.active_count) * Double(pageSize)
        let wired = Double(vmStats.wire_count) * Double(pageSize)
        
        let usedGB = (active + wired) / (1024 * 1024 * 1024)
        let availableGB = totalGB - usedGB
        
        return (usedGB, availableGB)
    }
    
    private func getCPUUsage() -> Double {
        var cpuInfo: processor_info_array_t?
        var numCpuInfo: mach_msg_type_number_t = 0
        var numCpus: natural_t = 0
        
        let result = host_processor_info(mach_host_self(),
                                         PROCESSOR_CPU_LOAD_INFO,
                                         &numCpus,
                                         &cpuInfo,
                                         &numCpuInfo)
        
        guard result == KERN_SUCCESS else { return 0 }
        
        var totalUsage: Double = 0
        
        for i in 0..<Int(numCpus) {
            let offset = Int(CPU_STATE_MAX) * i
            let user = cpuInfo![offset + Int(CPU_STATE_USER)]
            let system = cpuInfo![offset + Int(CPU_STATE_SYSTEM)]
            let idle = cpuInfo![offset + Int(CPU_STATE_IDLE)]
            
            let total = Double(user + system + idle)
            if total > 0 {
                totalUsage += Double(user + system) / total * 100
            }
        }
        
        if let cpuInfo = cpuInfo {
            vm_deallocate(mach_task_self_, vm_address_t(bitPattern: cpuInfo), vm_size_t(numCpuInfo) * vm_size_t(MemoryLayout<integer_t>.size))
        }
        
        return totalUsage / Double(numCpus)
    }
    
    private func calculatePressure(available: Double) -> String {
        if available < 2.0 { return "critical" }
        if available < 4.0 { return "warning" }
        return "normal"
    }
}

// ============================================================================
// 🛡️ SOVEREIGN SYMBIOSIS DASHBOARD
// ============================================================================

struct SovereignDashboard: View {
    @StateObject private var sniffer = TelemetrySniffer()
    @State private var resurrectionCount: Int = 0
    @State private var axiomHealth: Bool = true
    
    var body: some View {
        VStack(spacing: 0) {
            // Header
            headerView
            
            Divider()
            
            // Main metrics grid
            metricsGrid
            
            Divider()
            
            // Axiom status
            axiomStatusView
            
            Spacer()
            
            // Footer
            footerView
        }
        .frame(minWidth: 400, minHeight: 500)
        .background(Color.black.opacity(0.95))
        .onAppear {
            sniffer.startMonitoring()
        }
        .onDisappear {
            sniffer.stopMonitoring()
        }
    }
    
    // MARK: - Header
    
    var headerView: some View {
        HStack {
            Image(systemName: "brain.head.profile")
                .font(.title)
                .foregroundColor(.purple)
            
            Text("SOVEREIGN SYMBIOSIS")
                .font(.title2)
                .fontWeight(.bold)
                .foregroundColor(.white)
            
            Spacer()
            
            Text(sniffer.metrics.timestamp.prefix(19))
                .font(.caption)
                .foregroundColor(.gray)
        }
        .padding()
    }
    
    // MARK: - Metrics Grid
    
    var metricsGrid: some View {
        LazyVGrid(columns: [
            GridItem(.flexible()),
            GridItem(.flexible())
        ], spacing: 16) {
            // CPU
            MetricCard(
                title: "CPU",
                value: String(format: "%.1f%%", sniffer.metrics.cpu_usage),
                icon: "cpu",
                color: sniffer.metrics.cpu_usage > 85 ? .red : .green
            )
            
            // Memory
            MetricCard(
                title: "Memory",
                value: String(format: "%.1f GB free", sniffer.metrics.memory_available_gb),
                icon: "memorychip",
                color: sniffer.metrics.memory_available_gb < 2 ? .red : .green
            )
            
            // Thermal
            MetricCard(
                title: "Thermal",
                value: sniffer.metrics.thermal_state.uppercased(),
                icon: thermalIcon,
                color: thermalColor
            )
            
            // Resurrection
            MetricCard(
                title: "Resurrections",
                value: "\(resurrectionCount)",
                icon: "arrow.counterclockwise",
                color: .purple
            )
        }
        .padding()
    }
    
    var thermalIcon: String {
        switch sniffer.metrics.thermal_state {
        case "nominal": return "snowflake"
        case "fair": return "thermometer.medium"
        case "serious": return "flame"
        case "critical": return "flame.fill"
        default: return "thermometer"
        }
    }
    
    var thermalColor: Color {
        switch sniffer.metrics.thermal_state {
        case "nominal": return .blue
        case "fair": return .yellow
        case "serious": return .orange
        case "critical": return .red
        default: return .gray
        }
    }
    
    // MARK: - Axiom Status
    
    var axiomStatusView: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("AXIOM STATUS")
                .font(.headline)
                .foregroundColor(.gray)
            
            HStack(spacing: 20) {
                AxiomIndicator(name: "λ Love", healthy: true)
                AxiomIndicator(name: "σ Safety", healthy: true)
                AxiomIndicator(name: "α Abundance", healthy: !sniffer.metrics.abundance_violation)
                AxiomIndicator(name: "γ Growth", healthy: true)
            }
            
            if sniffer.metrics.abundance_violation {
                Text("⚠️ ABUNDANCE VIOLATION - Thermal Inversion Required")
                    .font(.caption)
                    .foregroundColor(.orange)
                    .padding(.top, 4)
            }
        }
        .padding()
        .background(Color.black.opacity(0.3))
    }
    
    // MARK: - Footer
    
    var footerView: some View {
        HStack {
            Circle()
                .fill(sniffer.metrics.abundance_violation ? Color.orange : Color.green)
                .frame(width: 10, height: 10)
            
            Text(sniffer.metrics.abundance_violation ? "INVERSION ACTIVE" : "SOVEREIGN")
                .font(.caption)
                .foregroundColor(.gray)
            
            Spacer()
            
            Text("🦢 Black Swan Labs")
                .font(.caption)
                .foregroundColor(.purple)
        }
        .padding()
    }
}

// MARK: - Components

struct MetricCard: View {
    let title: String
    let value: String
    let icon: String
    let color: Color
    
    var body: some View {
        VStack(spacing: 8) {
            Image(systemName: icon)
                .font(.title2)
                .foregroundColor(color)
            
            Text(value)
                .font(.title3)
                .fontWeight(.bold)
                .foregroundColor(.white)
            
            Text(title)
                .font(.caption)
                .foregroundColor(.gray)
        }
        .frame(maxWidth: .infinity)
        .padding()
        .background(Color.gray.opacity(0.1))
        .cornerRadius(12)
    }
}

struct AxiomIndicator: View {
    let name: String
    let healthy: Bool
    
    var body: some View {
        VStack(spacing: 4) {
            Image(systemName: healthy ? "shield.checkmark.fill" : "shield.slash.fill")
                .foregroundColor(healthy ? .green : .red)
            
            Text(name)
                .font(.caption2)
                .foregroundColor(.gray)
        }
    }
}

// MARK: - App Entry

@main
struct SovereignSymbiosisApp: App {
    var body: some Scene {
        WindowGroup {
            SovereignDashboard()
        }
        .windowStyle(.hiddenTitleBar)
    }
}
