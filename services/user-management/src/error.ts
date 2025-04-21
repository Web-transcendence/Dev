


export class MyError extends Error {
    code: number;
    constructor(message: string, code: number) {
        super(message);
        this.code = code;
    }
}

export class ServerError extends MyError {
    constructor(message: string, code: number) {
        super(`Server error: ${message}`, code);
    }
}

export class DataBaseError extends MyError {
    constructor(message: string, code: number) {
        super(`DataBase error: ${message}`, code);
    }
}

export class ConflictError extends MyError {
    constructor(message: string) {
        super(`Conflict error: ${message}`, 409);
    }
}

export class UnauthorizedError extends MyError {
    constructor(message: string) {
        super(`Unauthorized error: ${message}`, 401);
    }
}

export class InputError extends MyError {
    constructor(message: string) {
        super(`Input error: ${message}`, 400);

    }
}
