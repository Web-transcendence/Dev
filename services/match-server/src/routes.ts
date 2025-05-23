import {FastifyInstance, FastifyReply, FastifyRequest} from "fastify";
import {fetchIdByNickName, fetchMmrById, fetchNotifyUser} from "./utils.js";
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
import {
    generateRoom,
    joinRoom,
    leaveRoom, matchMaking, matchMakingUp, removeWaitingPlayer,
    startInviteMatch,
    startTournamentMatch,
    waitingList,
    waitingPlayer
} from "./netcode.js";
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
                    console.log("nickname:", data.nick);
                    player.dbId = await fetchIdByNickName(data.nick);
                    console.log("dbId:", player.dbId);
                    if (data.room)
                        room = msg.room;
                    if (room === -1){
                        player.mmr = await fetchMmrById(player.dbId);
                        console.log("mmr:", player.mmr);
                    }
                } catch (error) {
                    console.log(error);
                    return ;
                }
                init = true;
            } else if (msg.type === "input") {
                const {data, success, error} = inputSchema.safeParse(JSON.parse(message.toString()));
                if (!success || !data) {
                    console.error(error);
                    return ;
                }
                if (mode === "local" && (data.key === "w" || data.key === "s")) {
                    resetInput(player.input, "left");
                    inputHandler(data.key, data.state, player.input);
                } else if (mode === "local" && (data.key === "ArrowUp" || data.key === "ArrowDown")) {
                    resetInput(player.input, "right");
                    inputHandler(data.key, data.state, player.input);
                }
                else if (mode === "spec" && data.state === "down")
                    changeRoomSpec(player);
                else {
                    resetInput(player.input, "all");
                    inputHandler(data.key, data.state, player.input);
                }
            } else if (init && msg.type === "ready") {
                const {data, success, error} = readySchema.safeParse(JSON.parse(message.toString()));
                if (!success || !data) {
                    console.error(error);
                    return ;
                }
                mode = data.mode;
                switch (mode) {
                    case "local":
                        solo = true;
                        soloMode(player, solo);
                        break;
                    case "remote":
                        if (room === -1) {
                            waitingList.push(new waitingPlayer(player));
                            if (!matchMakingUp)
                                matchMaking();
                        }
                        else
                            joinRoom(player, room);
                        break;
                    case "spec":
                        joinRoomSpec(player, room);
                        break;
                    default:
                        break;
                }
            }
        });
        socket.on("close", () => {
            console.log(`id: ${userId}`);
            switch (mode) {
                case "local":
                    solo = false;
                    break;
                case "remote":
                    if (room === -1)
                        removeWaitingPlayer(player);
                    leaveRoom(userId);
                    break;
                case "spec":
                    leaveRoomSpec(userId);
                    break;
                default:
                    break;
            }
            console.log("Client disconnected");
        });
    });

    fastify.post('/generateRoom', async (req: FastifyRequest, res: FastifyReply) => {
        const roomId = generateRoom();
        return (res.status(200).send({roomId: roomId}));
    })

    fastify.post('/generateTournamentRoom', async (req: FastifyRequest, res: FastifyReply) => {
        const roomId = generateRoom("tournament");
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
            const params = req.params as { id: string }
            const oppId = Number(params.id)
            const id = Number(req.headers.id);

            const response = await fetch(`http://social:6500/checkFriend`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'authorization': `${INTERNAL_PASSWORD}`
                },
                body: JSON.stringify({id1: id, id2: oppId})
            })
            if (!response.ok) {
                res.status(409).send({message: `this user isn't in your friendlist`})
            }

            const roomID = await startInviteMatch(id, oppId);
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

            console.log(`match ${ids.id1} VS ${ids.id2} dans tournamentGame matchServer`)
            await startTournamentMatch(ids.id1, ids.id2);

            return res.status(200).send();
        } catch (err) {
            console.error(err)
            return res.status(500).send()
        }
    })

}
