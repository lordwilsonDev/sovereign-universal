#!/usr/bin/env python3
"""
🧬 PROJECT CONVERGER - Unified Neural Knowledge Graph
======================================================
Absorbs 300+ projects into a Local-First Sovereign Mind.

Features:
- Extraction: Pulls Axiomatic Core from each project
- Indexing: SQLite storage for zero-dependency convergence
- Cross-Pollination: Remember solutions across languages/time

"Your 9 months of study becoming an Autonomous Civilization."
"""

import os
import re
import ast
import json
import sqlite3
import hashlib
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any


# ============================================================================
# 📊 DATA STRUCTURES
# ============================================================================

@dataclass
class ProjectDNA:
    """The extracted 'spirit' of a project"""
    name: str
    path: str
    language: str
    core_logic: str           # Summary of what it does
    key_functions: List[str]  # Main functions/classes
    axiom_alignment: float    # 0-100 sovereignty score
    complexity: int           # Lines of code
    dependencies: List[str]   # External dependencies
    tags: List[str]           # Classification tags
    inversions: List[str]     # Past problem solutions
    last_active: str          # Last modification date


@dataclass
class KnowledgeNode:
    """A node in the knowledge graph"""
    id: int
    project_name: str
    concept: str
    implementation: str
    language: str
    axiom_score: float
    related_nodes: List[int]


# ============================================================================
# 🧬 PROJECT CONVERGER
# ============================================================================

class ProjectConverger:
    """
    The Project Converger - Unified Neural Knowledge Graph.
    
    Absorbs project DNA into a queryable Sovereign Mind.
    """
    
    def __init__(self, db_path: str = "sovereign_mind.db"):
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        self.conn.row_factory = sqlite3.Row
        self.create_schema()
    
    def create_schema(self):
        """Create the Knowledge Graph schema"""
        self.conn.executescript("""
            -- Project DNA: Core identity of each project
            CREATE TABLE IF NOT EXISTS project_dna (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                project_name TEXT UNIQUE,
                project_path TEXT,
                language TEXT,
                core_logic TEXT,
                key_functions TEXT,  -- JSON array
                axiom_alignment REAL,
                complexity INTEGER,
                dependencies TEXT,   -- JSON array
                tags TEXT,           -- JSON array
                inversions TEXT,     -- JSON array of solved problems
                last_active TIMESTAMP,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            
            -- Knowledge Nodes: Extracted concepts and implementations
            CREATE TABLE IF NOT EXISTS knowledge_nodes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                project_id INTEGER,
                concept TEXT,
                implementation TEXT,
                language TEXT,
                axiom_score REAL,
                tags TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (project_id) REFERENCES project_dna(id)
            );
            
            -- Cross References: Links between related concepts
            CREATE TABLE IF NOT EXISTS cross_references (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source_node INTEGER,
                target_node INTEGER,
                similarity REAL,
                relationship TEXT,
                FOREIGN KEY (source_node) REFERENCES knowledge_nodes(id),
                FOREIGN KEY (target_node) REFERENCES knowledge_nodes(id)
            );
            
            -- Inversions: Problem-solution pairs
            CREATE TABLE IF NOT EXISTS inversions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                project_id INTEGER,
                problem TEXT,
                solution TEXT,
                axiom_applied TEXT,
                success_score REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (project_id) REFERENCES project_dna(id)
            );
            
            -- Search index
            CREATE INDEX IF NOT EXISTS idx_dna_name ON project_dna(project_name);
            CREATE INDEX IF NOT EXISTS idx_dna_tags ON project_dna(tags);
            CREATE INDEX IF NOT EXISTS idx_nodes_concept ON knowledge_nodes(concept);
        """)
        self.conn.commit()
    
    def extract_project_dna(self, project_path: str) -> Optional[ProjectDNA]:
        """Extract DNA from a project directory"""
        path = Path(project_path)
        
        if not path.exists():
            return None
        
        # Detect project type and language
        language = self._detect_language(path)
        
        # Find key files
        key_files = self._find_key_files(path, language)
        
        # Extract core logic summary
        core_logic = self._extract_core_logic(key_files, language)
        
        # Extract function/class names
        key_functions = self._extract_key_functions(key_files, language)
        
        # Count complexity
        complexity = self._count_lines(path)
        
        # Extract dependencies
        dependencies = self._extract_dependencies(path, language)
        
        # Calculate axiom alignment (from auditor)
        axiom_score = self._calculate_axiom_alignment(key_files)
        
        # Generate tags
        tags = self._generate_tags(path, core_logic, dependencies)
        
        # Get last modification
        last_active = self._get_last_active(path)
        
        return ProjectDNA(
            name=path.name,
            path=str(path),
            language=language,
            core_logic=core_logic[:500],
            key_functions=key_functions[:20],
            axiom_alignment=axiom_score,
            complexity=complexity,
            dependencies=dependencies[:20],
            tags=tags,
            inversions=[],
            last_active=last_active
        )
    
    def _detect_language(self, path: Path) -> str:
        """Detect primary language of project"""
        extensions = {}
        
        for f in path.rglob('*'):
            if f.is_file():
                ext = f.suffix.lower()
                extensions[ext] = extensions.get(ext, 0) + 1
        
        lang_map = {
            '.py': 'python',
            '.swift': 'swift',
            '.rs': 'rust',
            '.ts': 'typescript',
            '.js': 'javascript',
            '.go': 'go',
            '.java': 'java',
            '.cpp': 'cpp',
            '.c': 'c',
        }
        
        for ext, lang in lang_map.items():
            if ext in extensions and extensions[ext] > 0:
                return lang
        
        return 'unknown'
    
    def _find_key_files(self, path: Path, language: str) -> List[Path]:
        """Find the most important files in a project"""
        key_files = []
        
        ext_map = {
            'python': ['.py'],
            'swift': ['.swift'],
            'rust': ['.rs'],
            'typescript': ['.ts', '.tsx'],
            'javascript': ['.js', '.jsx'],
        }
        
        extensions = ext_map.get(language, ['.py'])
        
        # Priority files
        priority_names = ['main', 'app', 'index', 'lib', 'core', 'controller', 'server']
        
        for ext in extensions:
            for f in path.rglob(f'*{ext}'):
                # Skip test files and node_modules
                if 'test' in str(f).lower() or 'node_modules' in str(f):
                    continue
                
                # Prioritize key files
                if any(p in f.stem.lower() for p in priority_names):
                    key_files.insert(0, f)
                else:
                    key_files.append(f)
        
        return key_files[:10]  # Top 10 files
    
    def _extract_core_logic(self, files: List[Path], language: str) -> str:
        """Extract summary of core logic"""
        summaries = []
        
        for f in files[:5]:
            try:
                content = f.read_text(errors='ignore')
                
                # Extract docstrings and comments
                if language == 'python':
                    # Get module docstring
                    match = re.search(r'^"""(.*?)"""', content, re.DOTALL)
                    if match:
                        summaries.append(match.group(1).strip()[:200])
                    
                    # Get class descriptions
                    classes = re.findall(r'class (\w+).*?:.*?"""(.*?)"""', content, re.DOTALL)
                    for name, doc in classes[:3]:
                        summaries.append(f"{name}: {doc.strip()[:100]}")
                
            except:
                pass
        
        return " | ".join(summaries)[:500] if summaries else "No documentation"
    
    def _extract_key_functions(self, files: List[Path], language: str) -> List[str]:
        """Extract key function and class names"""
        functions = []
        
        for f in files[:5]:
            try:
                content = f.read_text(errors='ignore')
                
                if language == 'python':
                    # Classes
                    classes = re.findall(r'^class (\w+)', content, re.MULTILINE)
                    functions.extend([f"class:{c}" for c in classes])
                    
                    # Functions
                    funcs = re.findall(r'^def (\w+)', content, re.MULTILINE)
                    functions.extend([f"def:{fn}" for fn in funcs if not fn.startswith('_')])
                
                elif language == 'swift':
                    classes = re.findall(r'^(?:class|struct) (\w+)', content, re.MULTILINE)
                    functions.extend(classes)
                    
            except:
                pass
        
        return list(set(functions))[:20]
    
    def _count_lines(self, path: Path) -> int:
        """Count total lines of code"""
        total = 0
        
        for f in path.rglob('*'):
            if f.is_file() and f.suffix in ['.py', '.swift', '.rs', '.ts', '.js']:
                try:
                    total += len(f.read_text().splitlines())
                except:
                    pass
        
        return total
    
    def _extract_dependencies(self, path: Path, language: str) -> List[str]:
        """Extract project dependencies"""
        deps = []
        
        if language == 'python':
            req_file = path / 'requirements.txt'
            if req_file.exists():
                deps = [l.split('==')[0].strip() for l in req_file.read_text().splitlines() if l.strip()]
        
        elif language == 'javascript' or language == 'typescript':
            pkg_file = path / 'package.json'
            if pkg_file.exists():
                try:
                    pkg = json.loads(pkg_file.read_text())
                    deps = list(pkg.get('dependencies', {}).keys())
                except:
                    pass
        
        return deps
    
    def _calculate_axiom_alignment(self, files: List[Path]) -> float:
        """Calculate axiom alignment score"""
        score = 50.0  # Base score
        
        for f in files:
            try:
                content = f.read_text(errors='ignore').lower()
                
                # Positive indicators
                if 'try' in content or 'except' in content:
                    score += 5
                if 'validate' in content or 'sanitize' in content:
                    score += 5
                if 'local' in content or 'offline' in content:
                    score += 10
                
                # Negative indicators
                if 'api.openai' in content or 'requests.post' in content:
                    score -= 20
                if 'eval(' in content or 'exec(' in content:
                    score -= 15
                    
            except:
                pass
        
        return max(0, min(100, score))
    
    def _generate_tags(self, path: Path, core_logic: str, deps: List[str]) -> List[str]:
        """Generate classification tags"""
        tags = []
        
        # From path name
        name = path.name.lower()
        if 'ai' in name or 'ml' in name:
            tags.append('ai')
        if 'web' in name or 'api' in name:
            tags.append('web')
        if 'cli' in name or 'tool' in name:
            tags.append('cli')
        
        # From dependencies
        if 'fastapi' in deps or 'flask' in deps:
            tags.append('backend')
        if 'react' in deps or 'vue' in deps:
            tags.append('frontend')
        if 'tensorflow' in deps or 'torch' in deps:
            tags.append('ml')
        
        return list(set(tags))
    
    def _get_last_active(self, path: Path) -> str:
        """Get last modification date"""
        latest = 0
        
        for f in path.rglob('*'):
            if f.is_file():
                try:
                    mtime = f.stat().st_mtime
                    if mtime > latest:
                        latest = mtime
                except:
                    pass
        
        if latest:
            return datetime.fromtimestamp(latest).isoformat()
        return datetime.now().isoformat()
    
    def absorb_project(self, dna: ProjectDNA) -> int:
        """Inject a project's spirit into the Unified Mind"""
        cursor = self.conn.execute("""
            INSERT OR REPLACE INTO project_dna (
                project_name, project_path, language, core_logic,
                key_functions, axiom_alignment, complexity,
                dependencies, tags, inversions, last_active
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            dna.name,
            dna.path,
            dna.language,
            dna.core_logic,
            json.dumps(dna.key_functions),
            dna.axiom_alignment,
            dna.complexity,
            json.dumps(dna.dependencies),
            json.dumps(dna.tags),
            json.dumps(dna.inversions),
            dna.last_active
        ))
        self.conn.commit()
        
        print(f"🧬 [Converger] {dna.name} has been synthesized into the Core.")
        return cursor.lastrowid
    
    def scan_and_absorb(self, root_dir: str) -> int:
        """Scan directory and absorb all projects"""
        print(f"""
╔══════════════════════════════════════════════════════════════╗
║  🧬 PROJECT CONVERGER                                        ║
║  Unified Neural Knowledge Graph                              ║
╚══════════════════════════════════════════════════════════════╝

Scanning: {root_dir}
""")
        
        root = Path(root_dir)
        absorbed = 0
        
        # Find project directories (containing key files)
        project_indicators = [
            'package.json', 'requirements.txt', 'Cargo.toml',
            'setup.py', 'pyproject.toml', 'go.mod'
        ]
        
        projects = set()
        
        for indicator in project_indicators:
            for f in root.rglob(indicator):
                projects.add(f.parent)
        
        # Also add direct subdirectories
        for d in root.iterdir():
            if d.is_dir() and not d.name.startswith('.'):
                projects.add(d)
        
        print(f"Found {len(projects)} potential projects\n")
        
        for project in sorted(projects):
            dna = self.extract_project_dna(str(project))
            if dna:
                self.absorb_project(dna)
                absorbed += 1
        
        print(f"\n✅ Absorbed {absorbed} projects into the Sovereign Mind")
        return absorbed
    
    def query(self, search_term: str) -> List[Dict]:
        """Search the knowledge graph"""
        cursor = self.conn.execute("""
            SELECT * FROM project_dna
            WHERE project_name LIKE ?
               OR core_logic LIKE ?
               OR tags LIKE ?
            ORDER BY axiom_alignment DESC
            LIMIT 10
        """, (f'%{search_term}%', f'%{search_term}%', f'%{search_term}%'))
        
        return [dict(row) for row in cursor.fetchall()]
    
    def cross_pollinate(self, concept: str) -> List[Dict]:
        """Find related implementations across projects"""
        results = self.query(concept)
        
        print(f"\n🔗 Cross-Pollination for '{concept}':")
        for r in results:
            print(f"   • {r['project_name']} ({r['language']}) - Score: {r['axiom_alignment']:.0f}")
        
        return results
    
    def get_stats(self) -> Dict:
        """Get statistics about the knowledge graph"""
        cursor = self.conn.execute("""
            SELECT 
                COUNT(*) as total_projects,
                AVG(axiom_alignment) as avg_alignment,
                SUM(complexity) as total_lines,
                COUNT(DISTINCT language) as languages
            FROM project_dna
        """)
        
        row = cursor.fetchone()
        return dict(row) if row else {}
    
    def close(self):
        """Close database connection"""
        self.conn.close()


# ============================================================================
# 🧪 CLI
# ============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Project Converger - Unified Knowledge Graph")
    parser.add_argument("path", nargs="?", default=".", help="Directory to scan")
    parser.add_argument("--query", "-q", type=str, help="Search the knowledge graph")
    parser.add_argument("--stats", action="store_true", help="Show graph statistics")
    parser.add_argument("--db", default="sovereign_mind.db", help="Database path")
    
    args = parser.parse_args()
    
    converger = ProjectConverger(args.db)
    
    if args.query:
        results = converger.cross_pollinate(args.query)
    elif args.stats:
        stats = converger.get_stats()
        print(f"\n📊 Knowledge Graph Statistics:")
        for k, v in stats.items():
            print(f"   {k}: {v}")
    else:
        converger.scan_and_absorb(args.path)
    
    converger.close()
