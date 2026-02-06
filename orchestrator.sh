#!/bin/bash

# CVAT Model Orchestrator Manager
# Mimics docker-compose interface (up, down, status, logs)

PID_FILE="orchestrator.pid"
LOG_FILE="orchestrator.log"
CMD="uv run python main.py"

function is_running() {
    if [ -f "$PID_FILE" ]; then
        pid=$(cat "$PID_FILE")
        if ps -p "$pid" > /dev/null; then
            return 0
        else
            rm "$PID_FILE"
            return 1
        fi
    fi
    return 1
}

function up() {
    if is_running; then
        echo "Orchestrator is already running (PID: $(cat $PID_FILE))."
        exit 0
    fi

    if [ "$1" == "-d" ]; then
        echo "Starting orchestrator in detached mode..."
        nohup $CMD > "$LOG_FILE" 2>&1 &
        echo $! > "$PID_FILE"
        echo "Orchestrator started in background (PID: $(cat $PID_FILE))."
        echo "Logs are being written to $LOG_FILE"
    else
        echo "Starting orchestrator in foreground..."
        $CMD
    fi
}

function down() {
    if is_running; then
        pid=$(cat "$PID_FILE")
        echo "Stopping orchestrator (PID: $pid)..."
        kill -TERM "$pid"
        
        # Wait for it to exit
        count=0
        while ps -p "$pid" > /dev/null && [ $count -lt 10 ]; do
            sleep 1
            ((count++))
        done
        
        if ps -p "$pid" > /dev/null; then
            echo "Force killing orchestrator..."
            kill -9 "$pid"
        fi
        
        rm "$PID_FILE"
        echo "Orchestrator stopped."
    else
        echo "Orchestrator is not running."
    fi
}

function status() {
    if is_running; then
        echo "Orchestrator is RUNNING (PID: $(cat $PID_FILE))."
    else
        echo "Orchestrator is STOPPED."
    fi
}

function logs() {
    if [ ! -f "$LOG_FILE" ]; then
        echo "No log file found ($LOG_FILE)."
        exit 1
    fi

    if [ "$1" == "-f" ]; then
        tail -f "$LOG_FILE"
    else
        cat "$LOG_FILE"
    fi
}

case "$1" in
    up)
        up "$2"
        ;;
    down)
        down
        ;;
    status)
        status
        ;;
    logs)
        logs "$2"
        ;;
    *)
        echo "Usage: $0 {up [-d]|down|status|logs [-f]}"
        exit 1
esac
