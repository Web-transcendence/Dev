import fastify, {FastifyReply, FastifyRequest} from 'fastify'
import {User} from "./User.js";
import {z} from "zod";
import sanitizeHtml from "sanitize-html";




const signUpSchema = z.object({
    name: z.string().min(3, "Minimum 3 caracteres"),
    email: z.string().email("Invalid email"),
    password: z.string().min(6, "Minimum 6 caracteres"),
});

const signInSchema = z.object({
    name: z.string().min(3, "Minimum 3 caracteres"),
    password: z.string().min(6, "Minimum 6 caracteres"),
});


const app = fastify();

app.post('/sign-up', async (req, res) => {
    try {
        console.log("sign-up");
        const zod_result = signUpSchema.safeParse(req.body);
        if (!zod_result.success)
            return res.status(400).send({json: zod_result.error.format()});
        let {name, email, password} = {name: sanitizeHtml(zod_result.data.name), email: sanitizeHtml(zod_result.data.email), password: sanitizeHtml(zod_result.data.password)};
        if (!name || !email || !password)
            return res.status(454).send({error: "All information are required !"});

        const client: User = new User(name);

        const addRes = await client.addClient(email, password);
        if (!addRes.success) {
            return res.status(409).send({error: `${addRes.result} already exists`});
        }
        return res.status(201).send({token: addRes.result, username: name , redirect: "post/login"});
    }
    catch(err) {
        return res.status(500).send({error: "Server error: ", err});
    }
});

app.post('/sign-in', async (req: FastifyRequest, res: FastifyReply) => {
    try {
        console.log("sign-in");
        const zod_result = signInSchema.safeParse(req.body);
        if (!zod_result.success)
            return res.status(400).send({json: zod_result.error.format()});
        let {name, password} = {
            name: sanitizeHtml(zod_result.data.name),
            password: sanitizeHtml(zod_result.data.password)
        };
        if (!name || !password)
            return res.status(454).send({error: "All information are required !"});

        const user = new User(name);
        if (user.getStatus() === "NotExists")
            return res.status(401).send({error: "Invalid username"});

        if (await user.isPasswordValid(password))
            return res.status(201).send({token: user.makeToken(), username: name , redirect: "post/login"});
        else
            return res.status(401).send({error: "Invalid password"});
    }
    catch (err) {
        res.status(500).send({error: "Server error: ", err});
    }
});


app.listen({port: 5000, host: '0.0.0.0'}, (err, adrr) => {
    if (err) {
        console.error(err);
        process.exit(1);
    }
    console.log(`server running on ${adrr}`)
})