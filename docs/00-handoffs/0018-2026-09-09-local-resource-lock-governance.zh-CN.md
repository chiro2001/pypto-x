# PyPTO-X 本机资源锁治理与重启恢复快照

归档序号：`0018`

归档日期：2026-09-09（Asia/Shanghai）

状态：`LOCAL_RESOURCE_LOCK_AND_RUNTIME_GUARD_ACTIVE`

## 触发原因

M1E 曾直接运行 Qwen `[1,6144,1024]` 的 Python heredoc lowering/compile，产生明显 CPU/内存压力。用户终止相关进程，随后本机异常重启。核对确认两个进程均属于 PyPTO-X 子任务；终止发生在 artifact 生成前，没有损坏源码、Git 对象或既有 smoke 日志。

重启后确认：

- `local FREE`、`gamepc FREE`，没有存活或不一致 owner；
- integration 仍 clean，HEAD `fe6b54a0f40e739d5ebed87baa5d608565171695`；
- M1E worktree 的未提交源码修改完整保留；
- 本机恢复为 12 online CPU、约 30 GiB 总内存与约 26 GiB `MemAvailable`。

## 生效规则

权威锁协议：

```text
/home/chiro/projects/.resource-locks/README.md
```

项目 heavy 入口：

```text
scripts/resource/run_local_heavy.sh
```

full pytest、多用例 QEMU、真实大 shape lowering/compile、并行构建，或预计使用至少 6/12 CPU、4 GiB 内存的命令，必须先取得全局 `local` 锁。返回 75/69 时等待，禁止裸跑、抢占或删除 guard/owner。

默认保护：启动至少 8 GiB `MemAvailable`，保留 4 GiB，最多 6 CPU；user cgroup 设置动态 `MemoryHigh/MemoryMax`、`MemorySwapMax=0` 与 CPU quota；supervisor 每 2 秒记录 `MemAvailable`、任务树 RSS/CPU、load 和 PSI，连续低内存时只停止本任务进程组。

包装器已验证锁取得/释放、CPU affinity、准入失败 69、cgroup 与采样日志。受控测试把 task `MemoryMax` 设为 256 MiB 后申请 400 MiB，结果仅该任务以 137 退出，主机保持正常且 `local` 自动回到 FREE。M1E 恢复后不得重跑已有 smoke；所有 heavy 测试和大 shape driver 必须使用新入口。
