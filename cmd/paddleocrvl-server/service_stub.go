//go:build !windows

package main

import "fmt"

func handleServiceCommand(args []string) (bool, error) {
	if len(args) == 0 || args[0] != "service" {
		return false, nil
	}
	return true, fmt.Errorf("service commands are only supported on Windows")
}
