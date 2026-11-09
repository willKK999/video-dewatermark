package control

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestMigrationFilesSorted(t *testing.T) {
	dir := t.TempDir()
	for _, name := range []string{"002_b.sql", "001_a.sql", "notes.txt", "skip.md"} {
		if err := os.WriteFile(filepath.Join(dir, name), []byte("select 1;\n"), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	if err := os.Mkdir(filepath.Join(dir, "nested.sql"), 0o755); err != nil {
		t.Fatal(err)
	}
	got, err := MigrationFiles(dir)
	if err != nil {
		t.Fatal(err)
	}
	if len(got) != 2 || !strings.HasSuffix(got[0], "001_a.sql") || !strings.HasSuffix(got[1], "002_b.sql") {
		t.Fatalf("order = %v", got)
	}
}
