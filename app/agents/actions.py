"""高风险动作构造。命令与 diff 由后端事实生成，不由 LLM 自由发挥。"""

from __future__ import annotations

from app.core.text import short_hash
from app.core.timeutil import minutes_between


def reverse_diff(diff: str) -> str:
    lines: list[str] = []
    for line in diff.splitlines():
        if line.startswith("+++") or line.startswith("---"):
            lines.append(line)
        elif line.startswith("+"):
            lines.append("-" + line[1:])
        elif line.startswith("-"):
            lines.append("+" + line[1:])
        else:
            lines.append(line)
    return "\n".join(lines)


def build_action(
    kind: str,
    *,
    service: str | None,
    evidence: list[dict],
    session_id: str,
    root_cause: str = "",
    anomaly_start: str | None = None,
) -> dict | None:
    if not service or kind == "none":
        return None

    change = None
    if kind == "rollback":
        preferred = "config" if root_cause == "config_change" else "deploy"
        # 回滚目标取“异常前最近一次”的同类变更，而不是窗口内任意一条
        candidates = []
        for item in evidence:
            if item.get("source") != "change":
                continue
            meta = item.get("metadata") or {}
            if meta.get("change_type") == preferred:
                ts = item.get("timestamp") or ""
                if anomaly_start and ts:
                    try:
                        if minutes_between(anomaly_start, ts) > 2:
                            continue  # 异常开始后才出现的变更通常是缓解措施
                    except Exception:
                        pass
                candidates.append((ts, {**meta, "title": item.get("title")}))
        if candidates:
            candidates.sort(key=lambda pair: pair[0], reverse=True)
            change = candidates[0][1]

    if kind == "rollback":
        if not change:
            return None
        change_id = change.get("change_id")
        change_type = change.get("change_type")
        diff = reverse_diff(change.get("diff", ""))
        if change_type == "config":
            command = (
                f"opsctl config rollback --service {service} --change {change_id} "
                "--to-previous --reason ops-diagnostic-agent"
            )
        else:
            command = (
                f"opsctl deploy rollback --service {service} --change {change_id} "
                "--to-previous --reason ops-diagnostic-agent"
            )
        target = change_id
        rationale = f"回滚变更 {change_id}，该变更在异常开始前发生且与证据时间线吻合。"
        risk = "high"
    elif kind == "scale":
        command = f"opsctl scale --service {service} --replicas +2 --reason ops-diagnostic-agent"
        target = service
        rationale = "扩容以缓解 CPU/排队压力；具体副本数需值班人员根据容量确认。"
        risk = "medium"
        diff = "- replicas: current\n+ replicas: current + 2"
    elif kind == "cleanup":
        command = (
            f"opsctl storage cleanup --service {service} --path /data/logs "
            "--older-than 7d --reason ops-diagnostic-agent"
        )
        target = "/data/logs"
        rationale = "清理过期日志释放磁盘空间；执行前需确认文件保留策略。"
        risk = "medium"
        diff = "- delete /data/logs/*.gz older than 7d\n+ free disk space"
    elif kind == "restart":
        command = f"opsctl rollout restart --service {service} --reason ops-diagnostic-agent"
        target = service
        rationale = "滚动重启以恢复进程状态；仅在其他证据指向进程内异常时使用。"
        risk = "high"
        diff = "- pod revision: current\n+ pod revision: new"
    else:
        return None

    idempotency_key = short_hash(
        f"{session_id}:{kind}:{service}:{target}", 24
    )
    return {
        "kind": kind,
        "target": target,
        "command": command,
        "diff": diff,
        "rationale": rationale,
        "risk": risk,
        "dry_run": True,
        "idempotency_key": idempotency_key,
    }