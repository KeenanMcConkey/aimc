"""Executable container formats, signing and the sandbox."""
from __future__ import annotations

import os
import shutil
import struct
import subprocess
import sys
import tempfile
import unittest

from aimc.codegen import Target, compile_module
from aimc.ir import Intrinsic, ModuleBuilder
from aimc.runtime import Sandbox, UnsupportedTarget, write_executable
from aimc.runtime.elf import write_elf
from aimc.runtime.macho import write_macho


def hello_module():
    mb = ModuleBuilder()
    f = mb.function("main", 0)
    f.write_str(b"hello\n")
    f.ret(f.const(5))
    return mb.build()


class ElfTests(unittest.TestCase):
    def test_header_fields(self):
        for arch, em in (("x86_64", 0x3E), ("aarch64", 0xB7)):
            img = compile_module(hello_module(), Target(arch, "linux"))
            data = write_elf(img)
            self.assertEqual(data[:4], b"\x7fELF")
            self.assertEqual(data[4], 2)                       # ELFCLASS64
            e_type, e_machine = struct.unpack_from("<HH", data, 16)
            self.assertEqual((e_type, e_machine), (2, em))
            e_entry, e_phoff = struct.unpack_from("<QQ", data, 24)
            self.assertEqual(e_entry, 0x400000 + 0x80 + img.start_offset)
            p_type, p_flags, p_offset, p_vaddr = struct.unpack_from("<IIQQ", data, e_phoff)
            self.assertEqual((p_type, p_flags, p_offset, p_vaddr), (1, 5, 0, 0x400000))
            self.assertEqual(data[0x80:0x80 + len(img.code)], img.code)

    @unittest.skipUnless(shutil.which("file"), "file(1) not available")
    def test_file_recognises_elf(self):
        img = compile_module(hello_module(), Target.parse("aarch64-linux"))
        with tempfile.NamedTemporaryFile(delete=False) as fh:
            fh.write(write_elf(img))
        try:
            out = subprocess.run(["file", fh.name], capture_output=True, text=True).stdout
        finally:
            os.unlink(fh.name)
        self.assertIn("ELF 64-bit", out)
        self.assertIn("ARM aarch64", out)

    def test_wrong_os_rejected(self):
        with self.assertRaises(ValueError):
            write_elf(compile_module(hello_module(), Target.parse("x86_64-macos")))


class MachOTests(unittest.TestCase):
    def test_header_fields(self):
        for arch, cputype in (("x86_64", 0x01000007), ("aarch64", 0x0100000C)):
            img = compile_module(hello_module(), Target(arch, "macos"))
            data = write_macho(img)
            magic, cpu, _, filetype, ncmds = struct.unpack_from("<IIIII", data, 0)
            self.assertEqual((magic, cpu, filetype, ncmds), (0xFEEDFACF, cputype, 2, 9))
            self.assertEqual(data[0x1000:0x1000 + len(img.code)], img.code)
            self.assertIn(b"/usr/lib/dyld\0", data)
            # signature superblob is present and hashes the file up to its own offset
            idx = data.rfind(b"\xfa\xde\x0c\xc0")
            self.assertGreater(idx, 0)
            self.assertEqual(data.rfind(b"\xfa\xde\x0c\x02"), idx + 20)

    @unittest.skipUnless(sys.platform == "darwin" and shutil.which("codesign"), "macOS codesign not available")
    def test_signature_validates(self):
        for arch in ("x86_64", "aarch64"):
            img = compile_module(hello_module(), Target(arch, "macos"))
            with tempfile.TemporaryDirectory() as tmp:
                path = os.path.join(tmp, "prog")
                with open(path, "wb") as fh:
                    fh.write(write_macho(img))
                r = subprocess.run(["codesign", "--verify", "--verbose=2", path], capture_output=True, text=True)
                self.assertEqual(r.returncode, 0, r.stderr)


class SandboxTests(unittest.TestCase):
    def setUp(self):
        self.sb = Sandbox(timeout=3.0)

    def runnable(self):
        return [(Target.parse(t), s) for t in ("x86_64-linux", "x86_64-macos", "aarch64-linux", "aarch64-macos")
                for s in self.sb.strategies(Target.parse(t))]

    def test_hello_everywhere(self):
        pairs = self.runnable()
        self.assertTrue(pairs)
        for target, strategy in pairs:
            with self.subTest(target=target.name, strategy=strategy):
                img = compile_module(hello_module(), target)
                res = self.sb.run(img, strategy=strategy)
                self.assertEqual((res.stdout, res.exit_code, res.strategy), (b"hello\n", 5, strategy))

    def test_timeout_is_enforced(self):
        mb = ModuleBuilder()
        f = mb.function("main", 0)
        loop = f.new_block()
        f.jmp(loop)
        f.switch_to(loop)
        f.jmp(loop)
        m = mb.build()
        target, strategy = self.runnable()[0]
        res = Sandbox(timeout=0.5).run(compile_module(m, target), strategy=strategy)
        self.assertTrue(res.timed_out)

    def test_unsupported_target(self):
        foreign = Target("x86_64", "linux" if sys.platform == "darwin" else "macos")
        with self.assertRaises(UnsupportedTarget):
            self.sb.run(compile_module(hello_module(), foreign))

    def test_write_executable_dispatch(self):
        for name in ("x86_64-linux", "aarch64-macos"):
            data = write_executable(compile_module(hello_module(), Target.parse(name)))
            self.assertIn(data[:4], (b"\x7fELF", b"\xcf\xfa\xed\xfe"))


if __name__ == "__main__":
    unittest.main()
