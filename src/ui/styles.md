<style>
@import url('https://fonts.googleapis.com/css2?family=Lexend:wght@400;700&display=swap');

#input-container {
    position: fixed;
    bottom: 0;
    width: 100%;
    padding: 10px;
    background-color: white;
    z-index: 100;
}

h1, h2 {
    font-family: 'Lexend', sans-serif;
    font-weight: bold;
    background: -webkit-linear-gradient(left, white, cyan, magenta, yellow);
    background: linear-gradient(to right, white, cyan, magenta, yellow);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    display: inline;
    font-size: 2em;
    background-size: 300% 300%;
    animation: gradient 10s ease infinite;
}

@keyframes gradient {
    0% {
        background-position: 0% 50%;
    }
    50% {
        background-position: 100% 50%;
    }
    100% {
        background-position: 0% 50%;
    }
}

.user-avatar {
    float: right;
    width: 40px;
    height: 40px;
    margin-left: 5px;
    margin-bottom: -10px;
    border-radius: 50%;
    object-fit: cover;
}

.bot-avatar {
    float: left;
    width: 40px;
    height: 40px;
    margin-right: 5px;
    border-radius: 50%;
    object-fit: cover;
}
</style>
