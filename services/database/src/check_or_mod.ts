import {FastifyReply, FastifyRequest} from "fastify";
import {Client_db} from "./clientBase.js";

const SearchInDb = (username: string): boolean => {
    const stmt = Client_db.prepare("SELECT COUNT(*) AS count FROM Client WHERE name = ?");
    const result = stmt.get(username) as { count: number };
    return result.count > 0;
};

interface decodedToken {
    id: string,
    name: string,
    iat: number,
    exp: number,
}

export async function finder(req: FastifyRequest, reply: FastifyReply) {
    const {token: {name}} = req.body as { token: decodedToken };
    if (!name)
        return reply.status(401).send({ success: false, message: "Username is required" });
    const value = SearchInDb(name);
    return reply.send({ value });
}