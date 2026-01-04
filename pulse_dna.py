#!/usr/bin/env python3
"""
🧬 PULSE DNA EXTRACTOR - Civilization Mind Genome Mapper
=========================================================
Maps the genome of your legacy projects using LLM analysis.

Extracts:
- Core Intent: Primary problem the project solves
- Legacy Gaps: Atmospheric dependencies, bloat, entropy
- Axiom Alignment: Love, Safety, Abundance, Growth scores
- Inversion Potential: How to transform for Sovereign Stack

Enables Semantic Resonance - cross-project pattern detection.

"We transition from Data to Wisdom."
"""

import os
import re
import json
import time
import sqlite3
import httpx
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any


# ============================================================================
# 🔧 CONFIGURATION
# ============================================================================

OLLAMA_URL = "http://localhost:11434"
DEFAULT_MODEL = "llama3.2:latest"

# DNA Extraction Prompt
DNA_EXTRACTION_PROMPT = """SYSTEM ROLE: You are the Sovereign Pulse DNA Extractor.
MISSION: Analyze the provided project and extract its "Genetic Blueprint."

PROJECT NAME: {project_name}
PROJECT PATH: {project_path}

CODE CONTEXT (first 100 lines of key files):
```
{context_snippet}
```

FILE STRUCTURE:
{file_structure}

EXTRACTION FIELDS:

1. Core Intent: What was the primary problem this project attempted to solve?

2. Legacy Gaps: Identify any "Atmospheric" dependencies (Cloud APIs, bloat, high entropy).
   - External API calls (OpenAI, Google, etc.)
   - Missing error handling
   - Security vulnerabilities
   - Bloated dependencies

3. Axiom Alignment: Rate the project (1-10) on each:
   - Love: Does it serve human flourishing?
   - Safety: Is it secure and safe?
   - Abundance: Is it efficient, minimal waste?
   - Growth: Does it enable learning/evolution?

4. Inversion Potential: How can this logic be inverted to serve the current Sovereign Stack?

CONSTRAINT: Respond ONLY in this exact JSON format:
{{
    "core_intent": "string describing the main purpose",
    "legacy_gaps": ["gap1", "gap2", ...],
    "axiom_alignment": {{
        "love": 1-10,
        "safety": 1-10,
        "abundance": 1-10,
        "growth": 1-10
    }},
    "inversion_potential": "string describing transformation opportunity",
    "tags": ["tag1", "tag2", ...],
    "key_patterns": ["pattern1", "pattern2", ...]
}}

Be ruthless in identifying architectural debt. Output ONLY valid JSON."""


# ============================================================================
# 📊 DATA STRUCTURES
# ============================================================================

@dataclass
class ProjectDNA:
    """Extracted DNA of a project"""
    name: str
    path: str
    core_intent: str
    legacy_gaps: List[str]
    axiom_alignment: Dict[str, int]
    inversion_potential: str
    tags: List[str]
    key_patterns: List[str]
    extraction_date: str = field(default_factory=lambda: datetime.now().isoformat())
    
    @property
    def total_axiom_score(self) -> float:
        """Calculate total axiom alignment score"""
        if not self.axiom_alignment:
            return 0.0
        return sum(self.axiom_alignment.values()) / 4
    
    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "path": self.path,
            "core_intent": self.core_intent,
            "legacy_gaps": self.legacy_gaps,
            "axiom_alignment": self.axiom_alignment,
            "inversion_potential": self.inversion_potential,
            "tags": self.tags,
            "key_patterns": self.key_patterns,
            "total_score": self.total_axiom_score,
            "extraction_date": self.extraction_date
        }


# ============================================================================
# 🧬 PULSE DNA EXTRACTOR
# ============================================================================

class PulseDNAExtractor:
    """
    The Pulse DNA Extractor - Maps your project genome using LLM.
    
    Enables Semantic Resonance across 300+ projects.
    """
    
    def __init__(
        self,
        db_path: str = "sovereign_memory.db",
        model: str = DEFAULT_MODEL
    ):
        self.db_path = db_path
        self.model = model
        self.conn = sqlite3.connect(db_path)
        self.conn.row_factory = sqlite3.Row
        self._create_schema()
        
        self._extracted_count = 0
        self._failed_count = 0
    
    def _create_schema(self):
        """Create the DNA storage schema"""
        self.conn.executescript("""
            -- Project DNA Table
            CREATE TABLE IF NOT EXISTS project_dna (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                project_name TEXT UNIQUE,
                project_path TEXT,
                core_intent TEXT,
                legacy_gaps TEXT,          -- JSON array
                axiom_love INTEGER,
                axiom_safety INTEGER,
                axiom_abundance INTEGER,
                axiom_growth INTEGER,
                total_score REAL,
                inversion_potential TEXT,
                tags TEXT,                  -- JSON array
                key_patterns TEXT,          -- JSON array
                extraction_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            
            -- Semantic Resonance Index
            CREATE TABLE IF NOT EXISTS semantic_patterns (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                pattern TEXT UNIQUE,
                project_ids TEXT,           -- JSON array of project IDs
                frequency INTEGER DEFAULT 1,
                category TEXT
            );
            
            -- Neural Links (cross-project connections)
            CREATE TABLE IF NOT EXISTS neural_links (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source_project TEXT,
                target_project TEXT,
                resonance_score REAL,
                shared_patterns TEXT,       -- JSON array
                link_type TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            
            -- Indexes
            CREATE INDEX IF NOT EXISTS idx_dna_name ON project_dna(project_name);
            CREATE INDEX IF NOT EXISTS idx_dna_score ON project_dna(total_score);
            CREATE INDEX IF NOT EXISTS idx_patterns ON semantic_patterns(pattern);
        """)
        self.conn.commit()
    
    def get_context_snippet(self, project_path: str, max_lines: int = 100) -> str:
        """Get code context from key files"""
        path = Path(project_path)
        snippets = []
        
        # Priority files
        priority_names = ['main', 'app', 'index', 'lib', 'core', 'controller', 'server', '__init__']
        extensions = ['.py', '.swift', '.rs', '.ts', '.js', '.go']
        
        files_found = []
        
        # Find key files
        for ext in extensions:
            for f in path.rglob(f'*{ext}'):
                if 'node_modules' in str(f) or '__pycache__' in str(f):
                    continue
                if any(p in f.stem.lower() for p in priority_names):
                    files_found.insert(0, f)
                else:
                    files_found.append(f)
        
        # Extract snippets
        lines_collected = 0
        for f in files_found[:5]:  # Top 5 files
            if lines_collected >= max_lines:
                break
            
            try:
                content = f.read_text(errors='ignore')
                file_lines = content.splitlines()[:30]  # First 30 lines per file
                
                snippets.append(f"=== {f.name} ===")
                snippets.extend(file_lines)
                snippets.append("")
                
                lines_collected += len(file_lines)
            except:
                pass
        
        return "\n".join(snippets)
    
    def get_file_structure(self, project_path: str) -> str:
        """Get project file structure"""
        path = Path(project_path)
        structure = []
        
        for f in sorted(path.rglob('*'))[:30]:
            if 'node_modules' in str(f) or '__pycache__' in str(f):
                continue
            
            relative = f.relative_to(path)
            prefix = "📁" if f.is_dir() else "📄"
            structure.append(f"{prefix} {relative}")
        
        return "\n".join(structure)
    
    def analyze_project(self, project_name: str, project_path: str) -> Optional[ProjectDNA]:
        """Analyze a project using LLM"""
        context = self.get_context_snippet(project_path)
        structure = self.get_file_structure(project_path)
        
        if not context.strip():
            return None
        
        prompt = DNA_EXTRACTION_PROMPT.format(
            project_name=project_name,
            project_path=project_path,
            context_snippet=context[:3000],
            file_structure=structure
        )
        
        try:
            with httpx.Client(timeout=120.0) as client:
                response = client.post(
                    f"{OLLAMA_URL}/api/generate",
                    json={
                        "model": self.model,
                        "prompt": prompt,
                        "stream": False,
                        "options": {"temperature": 0.3}
                    }
                )
                
                if response.status_code == 200:
                    result = response.json().get("response", "")
                    return self._parse_dna(project_name, project_path, result)
                    
        except Exception as e:
            print(f"   ⚠️ LLM error: {e}")
        
        return None
    
    def _parse_dna(self, name: str, path: str, llm_response: str) -> Optional[ProjectDNA]:
        """Parse LLM response into ProjectDNA"""
        try:
            # Extract JSON from response
            json_match = re.search(r'\{[\s\S]*\}', llm_response)
            if not json_match:
                return None
            
            data = json.loads(json_match.group())
            
            axiom = data.get('axiom_alignment', {})
            
            return ProjectDNA(
                name=name,
                path=path,
                core_intent=data.get('core_intent', 'Unknown'),
                legacy_gaps=data.get('legacy_gaps', []),
                axiom_alignment={
                    'love': axiom.get('love', 5),
                    'safety': axiom.get('safety', 5),
                    'abundance': axiom.get('abundance', 5),
                    'growth': axiom.get('growth', 5)
                },
                inversion_potential=data.get('inversion_potential', 'None identified'),
                tags=data.get('tags', []),
                key_patterns=data.get('key_patterns', [])
            )
            
        except json.JSONDecodeError:
            return None
    
    def store_dna(self, dna: ProjectDNA) -> int:
        """Store extracted DNA in database"""
        cursor = self.conn.execute("""
            INSERT OR REPLACE INTO project_dna (
                project_name, project_path, core_intent, legacy_gaps,
                axiom_love, axiom_safety, axiom_abundance, axiom_growth,
                total_score, inversion_potential, tags, key_patterns
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            dna.name,
            dna.path,
            dna.core_intent,
            json.dumps(dna.legacy_gaps),
            dna.axiom_alignment.get('love', 5),
            dna.axiom_alignment.get('safety', 5),
            dna.axiom_alignment.get('abundance', 5),
            dna.axiom_alignment.get('growth', 5),
            dna.total_axiom_score,
            dna.inversion_potential,
            json.dumps(dna.tags),
            json.dumps(dna.key_patterns)
        ))
        self.conn.commit()
        
        # Update semantic patterns
        self._update_patterns(dna)
        
        return cursor.lastrowid
    
    def _update_patterns(self, dna: ProjectDNA):
        """Update semantic pattern index"""
        for pattern in dna.key_patterns:
            self.conn.execute("""
                INSERT INTO semantic_patterns (pattern, project_ids, frequency)
                VALUES (?, ?, 1)
                ON CONFLICT(pattern) DO UPDATE SET
                    project_ids = project_ids || ',' || ?,
                    frequency = frequency + 1
            """, (pattern, dna.name, dna.name))
        self.conn.commit()
    
    def extract_all(self, repo_path: str, limit: int = None) -> int:
        """Extract DNA from all projects in a directory"""
        print(f"""
╔══════════════════════════════════════════════════════════════╗
║  🧬 PULSE DNA EXTRACTOR                                      ║
║  Mapping the Genome of Your Legacy                           ║
╚══════════════════════════════════════════════════════════════╝

Repository: {repo_path}
Model: {self.model}
""")
        
        repo = Path(repo_path)
        projects = []
        
        # Find project directories
        for item in repo.iterdir():
            if item.is_dir() and not item.name.startswith('.'):
                projects.append(item)
        
        if limit:
            projects = projects[:limit]
        
        print(f"Found {len(projects)} projects to analyze\n")
        
        for i, project in enumerate(projects):
            print(f"[{i+1}/{len(projects)}] Analyzing: {project.name}")
            
            dna = self.analyze_project(project.name, str(project))
            
            if dna:
                self.store_dna(dna)
                self._extracted_count += 1
                print(f"   ✅ DNA extracted (Score: {dna.total_axiom_score:.1f})")
                print(f"   └─ {dna.core_intent[:60]}...")
            else:
                self._failed_count += 1
                print(f"   ⚠️ Extraction failed - manual review needed")
            
            # Brief pause to avoid overwhelming Ollama
            time.sleep(0.5)
        
        print(f"\n✅ Extraction complete: {self._extracted_count} digitized, {self._failed_count} failed")
        return self._extracted_count
    
    def find_resonance(self, concept: str) -> List[Dict]:
        """Find semantic resonance across projects"""
        cursor = self.conn.execute("""
            SELECT * FROM project_dna
            WHERE core_intent LIKE ?
               OR tags LIKE ?
               OR key_patterns LIKE ?
               OR inversion_potential LIKE ?
            ORDER BY total_score DESC
            LIMIT 10
        """, (f'%{concept}%', f'%{concept}%', f'%{concept}%', f'%{concept}%'))
        
        results = [dict(row) for row in cursor.fetchall()]
        
        if results:
            print(f"\n🔮 SEMANTIC RESONANCE for '{concept}':")
            for r in results:
                print(f"   • {r['project_name']} (Score: {r['total_score']:.1f})")
                print(f"     └─ {r['core_intent'][:50]}...")
        
        return results
    
    def suggest_inversion(self, current_file: str) -> Optional[Dict]:
        """Suggest inversions based on current work"""
        # Read current file
        try:
            content = Path(current_file).read_text()[:1000]
        except:
            return None
        
        # Find matching patterns in database
        keywords = re.findall(r'\b[A-Za-z_]+\b', content)[:20]
        
        for keyword in keywords:
            if len(keyword) > 4:  # Skip short words
                results = self.find_resonance(keyword)
                if results:
                    best_match = results[0]
                    
                    # Check for atmospheric gaps
                    gaps = json.loads(best_match.get('legacy_gaps', '[]'))
                    if gaps:
                        return {
                            "resonating_project": best_match['project_name'],
                            "legacy_gaps": gaps,
                            "inversion_suggestion": best_match['inversion_potential'],
                            "message": f"I detect a pattern from '{best_match['project_name']}'. "
                                      f"Consider applying the Local-First protocol."
                        }
        
        return None
    
    def get_stats(self) -> Dict:
        """Get extraction statistics"""
        cursor = self.conn.execute("""
            SELECT 
                COUNT(*) as total_projects,
                AVG(total_score) as avg_score,
                SUM(CASE WHEN total_score >= 7 THEN 1 ELSE 0 END) as sovereign_count,
                SUM(CASE WHEN total_score < 5 THEN 1 ELSE 0 END) as legacy_count
            FROM project_dna
        """)
        
        row = cursor.fetchone()
        
        return {
            "total_projects": row['total_projects'] or 0,
            "avg_score": row['avg_score'] or 0,
            "sovereign_count": row['sovereign_count'] or 0,
            "legacy_count": row['legacy_count'] or 0
        }
    
    def close(self):
        """Close database connection"""
        self.conn.close()


# ============================================================================
# 🔮 SEMANTIC RESONANCE ENGINE
# ============================================================================

class SemanticResonanceEngine:
    """
    Semantic Resonance - Cross-project pattern detection.
    
    "Lord Wilson, I detect a logic pattern in this file that resonates
    with Project #142 from December 2025..."
    """
    
    def __init__(self, db_path: str = "sovereign_memory.db"):
        self.conn = sqlite3.connect(db_path)
        self.conn.row_factory = sqlite3.Row
    
    def detect_resonance(self, code_snippet: str) -> List[Dict]:
        """Detect resonating projects from code snippet"""
        # Extract keywords from code
        keywords = set(re.findall(r'\b[A-Za-z_]{5,}\b', code_snippet))
        
        resonating = []
        
        for keyword in list(keywords)[:10]:
            cursor = self.conn.execute("""
                SELECT project_name, core_intent, total_score, legacy_gaps
                FROM project_dna
                WHERE core_intent LIKE ? 
                   OR key_patterns LIKE ?
                   OR tags LIKE ?
            """, (f'%{keyword}%', f'%{keyword}%', f'%{keyword}%'))
            
            for row in cursor.fetchall():
                resonating.append({
                    "project": row['project_name'],
                    "intent": row['core_intent'],
                    "score": row['total_score'],
                    "gaps": json.loads(row['legacy_gaps'] or '[]'),
                    "matched_on": keyword
                })
        
        # Deduplicate and sort
        seen = set()
        unique = []
        for r in resonating:
            if r['project'] not in seen:
                seen.add(r['project'])
                unique.append(r)
        
        return sorted(unique, key=lambda x: -x['score'])[:5]
    
    def generate_advice(self, resonances: List[Dict]) -> str:
        """Generate advisory message based on resonances"""
        if not resonances:
            return "No historical patterns detected."
        
        top = resonances[0]
        
        message = f"Lord Wilson, I detect a logic pattern that resonates with '{top['project']}' "
        
        if top['gaps']:
            message += f"(which had {len(top['gaps'])} legacy gaps: {', '.join(top['gaps'][:2])}). "
            message += "I recommend inverting that logic using the Local-First protocol."
        else:
            message += f"(Score: {top['score']}/10). This was Sovereign code - consider reusing."
        
        return message


# ============================================================================
# 🧪 CLI
# ============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Pulse DNA Extractor")
    parser.add_argument("path", nargs="?", default=".", help="Repository to scan")
    parser.add_argument("--limit", type=int, help="Limit number of projects")
    parser.add_argument("--db", default="sovereign_memory.db", help="Database path")
    parser.add_argument("--model", default="llama3.2:latest", help="LLM model")
    parser.add_argument("--resonance", "-r", type=str, help="Find semantic resonance")
    parser.add_argument("--stats", action="store_true", help="Show statistics")
    
    args = parser.parse_args()
    
    extractor = PulseDNAExtractor(db_path=args.db, model=args.model)
    
    if args.resonance:
        extractor.find_resonance(args.resonance)
    elif args.stats:
        stats = extractor.get_stats()
        print(f"\n📊 Civilization Mind Statistics:")
        for k, v in stats.items():
            print(f"   {k}: {v}")
    else:
        extractor.extract_all(args.path, limit=args.limit)
    
    extractor.close()
