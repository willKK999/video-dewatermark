package main

import (
	"context"
	"encoding/json"
	"github.com/jackc/pgx/v5/pgxpool"
	"github.com/minio/minio-go/v7"
	"github.com/minio/minio-go/v7/pkg/credentials"
	"github.com/redis/go-redis/v9"
	"log/slog"
	"net/http"
	"os"
	"os/signal"
	"strconv"
	"syscall"
	"time"
	"watermark/internal/control"
)

func env(k, d string) string {
	if v := os.Getenv(k); v != "" {
		return v
	}
	return d
}
func main() {
	ctx, stop := signal.NotifyContext(context.Background(), syscall.SIGTERM, syscall.SIGINT)
	defer stop()
	db, err := pgxpool.New(ctx, env("DATABASE_URL", "postgres://watermark:watermark@localhost:5432/watermark?sslmode=disable"))
	must(err)
	defer db.Close()
	must(db.Ping(ctx))
	if env("AUTO_MIGRATE", "false") == "true" {
		must(control.ApplyMigrations(ctx, db, "migrations"))
	}
	tokens := map[string]string{}
	must(json.Unmarshal([]byte(os.Getenv("API_TOKENS")), &tokens))
	if len(tokens) == 0 {
		panic("API_TOKENS must contain token-to-tenant mappings")
	}
	for token, tenant := range tokens {
		if len(token) < 16 || tenant == "" {
			panic("tokens require >=16 characters and a tenant")
		}
		for _, c := range tenant {
			if !(c >= 'a' && c <= 'z' || c >= 'A' && c <= 'Z' || c >= '0' && c <= '9' || c == '_' || c == '-') {
				panic("invalid tenant id")
			}
		}
	}
	creds := credentials.NewStaticV4(os.Getenv("S3_ACCESS_KEY"), os.Getenv("S3_SECRET_KEY"), "")
	opts := &minio.Options{Creds: creds, Secure: env("S3_SECURE", "false") == "true", Region: env("S3_REGION", "us-east-1")}
	store, err := minio.New(env("S3_ENDPOINT", "localhost:9000"), opts)
	must(err)
	signer, err := minio.New(env("S3_PUBLIC_ENDPOINT", env("S3_ENDPOINT", "localhost:9000")), opts)
	must(err)
	ro, err := redis.ParseURL(env("REDIS_URL", "redis://localhost:6379/0"))
	must(err)
	rd := redis.NewClient(ro)
	defer rd.Close()
	must(rd.Ping(ctx).Err())
	maxPending, e := strconv.Atoi(env("MAX_PENDING_JOBS", "10"))
	must(e)
	if maxPending < 1 {
		panic("MAX_PENDING_JOBS must be positive")
	}
	srv := &control.Server{DisableObjectTagging: env("S3_OBJECT_TAGGING", "true") != "true", MaxPendingJobs: maxPending, DB: db, Redis: rd, Store: store, Signer: signer, Bucket: env("S3_BUCKET", "watermark"), Tokens: tokens, Pipeline: env("PIPELINE_VERSION", "mvp-1")}
	go func() {
		tick := time.NewTicker(time.Second)
		defer tick.Stop()
		for {
			select {
			case <-ctx.Done():
				return
			case <-tick.C:
				c, cancel := context.WithTimeout(ctx, 10*time.Second)
				if e := srv.Reap(c); e != nil {
					slog.Error("lease sweep failed")
				}
				if e := srv.Tick(c); e != nil {
					slog.Error("outbox dispatch failed")
				}
				cancel()
			}
		}
	}()
	httpServer := &http.Server{Addr: env("API_ADDR", ":8080"), Handler: srv.Handler(), ReadHeaderTimeout: 5 * time.Second, ReadTimeout: 10 * time.Minute, WriteTimeout: 10 * time.Minute, IdleTimeout: 60 * time.Second}
	go func() {
		<-ctx.Done()
		c, cancel := context.WithTimeout(context.Background(), 10*time.Second)
		defer cancel()
		_ = httpServer.Shutdown(c)
	}()
	slog.Info("API listening", "addr", httpServer.Addr, "pipeline", srv.Pipeline)
	if e := httpServer.ListenAndServe(); e != nil && e != http.ErrServerClosed {
		panic(e)
	}
}
func must(e error) {
	if e != nil {
		panic(e)
	}
}
