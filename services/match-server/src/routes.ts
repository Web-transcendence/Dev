import {FastifyInstance, FastifyReply, FastifyRequest} from "fastify";
import {fetchIdByNickName, fetchNotifyUser} from "./utils.js";
import {
    generateId,
    initSchema,
    inputHandler,
    inputSchema,
    INTERNAL_PASSWORD,
    Player,
    readySchema,
    resetInput,
} from "./api.js";
import {getMatchHistory, MatchResult} from "./database.js"
import {soloMode} from "./solo.js";
import {generateRoom, joinRoom, leaveRoom, startInviteMatch, startTournamentMatch} from "./netcode.js";
import {z} from "zod";
import {changeRoomSpec, joinRoomSpec, leaveRoomSpec} from "./spectator.js";


const internalVerification = async (req, res) => {
    if (req.headers.authorization !== INTERNAL_PASSWORD)
        throw new Error(`only server can reach this endpoint`)
}

export default async function pongRoutes(fastify: FastifyInstance) {
    fastify.get('/ws', { websocket: true }, (socket, req) => {
        console.log("Client connected");
        const userId = generateId();
        const player = new Player(userId, socket);
        let init = false;
        let room = -1;
        let solo: boolean;
        let mode = "local";
        socket.on("message", async (message) => {
            const msg = JSON.parse(message.toString());
            if (!init && msg.type === "socketInit") {
                const {data, success, error} = initSchema.safeParse(JSON.parse(message.toString()));
                if (!success || !data) {
                    console.error(error);
                    return ;
                }
                player.name = data.nick;
                player.paddle.name = data.nick;
                if (player.name === "AI") {
                    player.frequency = 1000;
                    player.dbId = -2;
                }
                try {
                    player.dbId = await fetchIdByNickName(data.nick);
                } catch (error) {
                    console.log(error);
                }
                if (data.room)
                    room = msg.room;
                init = true;
            } else if (msg.type === "input") {
                const {data, success, error} = inputSchema.safeParse(JSON.parse(message.toString()));
                if (!success || !data) {
                    console.error(error);
                    return ;
                }
                if (mode !== "spec") {
                    resetInput(player.input);
                    inputHandler(data.key, data.state, player.input);
                } else if (data.state === "down")
                    changeRoomSpec(player);
            } else if (init && msg.type === "ready") {
                const {data, success, error} = readySchema.safeParse(JSON.parse(message.toString()));
                if (!success || !data) {
                    console.error(error);
                    return ;
                }
                mode = data.mode;
                if (data.mode === "remote")
                    joinRoom(player, room);
                else if (data.mode === "local") {
                    solo = true;
                    soloMode(player, solo);
                } else if (data.mode === "spec") {
                    joinRoomSpec(player, room);
                }
            }
        });
        socket.on("close", () => {
            if (mode === "local")
                solo = false;
            else if (mode === "remote")
                leaveRoom(userId);
            else if (mode === "spec")
                leaveRoomSpec(userId);
            console.log("Client disconnected");
        });
    });

    fastify.post('/generateRoom', async (req: FastifyRequest, res: FastifyReply) => {
        const roomId = generateRoom();
        return (res.status(200).send({roomId: roomId}));
    })

    fastify.get('/getMatchHistory', async (req: FastifyRequest, res: FastifyReply) => {
        try {
            const id = Number(req.headers.id);
            const MatchResult: MatchResult[] = getMatchHistory(id);
            return (res.status(200).send(MatchResult));
        } catch (error) {
            console.log(error);
            return (res.status(500).send({error}));
        }
    })

    fastify.get('/invitationGame/:id', async (req: FastifyRequest, res: FastifyReply) => {
        try {
            const id: number = Number(req.headers.id)
            const stringId = req.params as { id: string };
            const friendId = Number(stringId.id);
            if (!friendId)
                throw new Error("id must be a number");
            const response = fetch(`http://social:6500/checkFriend`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'authorization': `${INTERNAL_PASSWORD}`
                },
                body: JSON.stringify({id1: id, id2: friendId})
            })

            const roomID = await startInviteMatch(id, friendId);
            return res.status(200).send({roomId: roomID});
        } catch (err) {
            console.error(err)
            return res.status(400).send(err)
        }
    })

    fastify.post('/tournamentGame', {preHandler: internalVerification}, async (req: FastifyRequest, res: FastifyReply) => {
        try {

            const ids = z.object({
                id1: z.number(),
                id2: z.number()
            }).parse(req.body)

            const winnerId = await startTournamentMatch(ids.id1, ids.id2);

            return res.status(200).send({id: winnerId});
        } catch (err) {
            console.error(err)
            return res.status(500).send()
        }
    })
}
