


export class MyError extends Error {
    code: number;
    toSend: string;

    constructor(log: string, toSend: string, code: number) {
        super(log);
        this.toSend = toSend;
        this.code = code;
    }
}

export class ServerError extends MyError {
    constructor(log: string, code: number) {
        super(`Server error: ${log}`, `error 500: internal error system`, code);
    }
}

export class NotFoundError extends MyError {
    constructor(log: string, toSend: string) {
        super(`Data doesn't exist: ${log}`, toSend, 404);
    }
}

export class ConflictError extends MyError {
    constructor(log: string, toSend: string) {
        super(`Conflict error: ${log}`, toSend, 409);
    }
}

export class UnauthorizedError extends MyError {
    constructor(log: string, toSend: string) {
        super(`Unauthorized error: ${log}`, toSend, 401);
    }
}

export class InputError extends MyError {
    constructor(log: string, toSend: string) {
        super(`Input error: ${log}`,toSend, 400);
    }
}
