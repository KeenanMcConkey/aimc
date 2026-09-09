"""aimc - an intent-to-machine-code compiler.

Pipeline:  natural language  ->  IR (aimc.ir)  ->  raw machine code (aimc.codegen)
           ->  executable image / JIT (aimc.runtime)  ->  verified execution (aimc.verify)
"""
__version__ = "0.1.0"
