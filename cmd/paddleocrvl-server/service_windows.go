//go:build windows

package main

import (
	"errors"
	"fmt"
	"os"
	"strings"
	"time"

	"golang.org/x/sys/windows"
	"golang.org/x/sys/windows/svc"
	"golang.org/x/sys/windows/svc/eventlog"
	"golang.org/x/sys/windows/svc/mgr"
)

const (
	serviceName        = "PaddleOCRVLService"
	serviceDisplayName = "PaddleOCR-VL Inference Service"
	serviceDescription = "Local PaddleOCR-VL inference HTTP service."
)

func handleServiceCommand(args []string) (bool, error) {
	if len(args) == 0 || args[0] != "service" {
		return false, nil
	}
	if len(args) < 2 {
		return true, fmt.Errorf("usage: %s service <install|uninstall|start|stop|run> [server flags]", os.Args[0])
	}
	action := strings.ToLower(args[1])
	serverArgs := args[2:]
	switch action {
	case "install":
		return true, installWindowsService(serverArgs)
	case "uninstall", "remove":
		return true, uninstallWindowsService()
	case "start":
		return true, startWindowsService()
	case "stop":
		return true, stopWindowsService()
	case "run":
		return true, runWindowsService(serverArgs)
	default:
		return true, fmt.Errorf("unknown service action %q", args[1])
	}
}

func installWindowsService(serverArgs []string) error {
	exe, err := os.Executable()
	if err != nil {
		return err
	}
	m, err := mgr.Connect()
	if err != nil {
		return err
	}
	defer m.Disconnect()
	if existing, err := m.OpenService(serviceName); err == nil {
		existing.Close()
		if err := uninstallWindowsService(); err != nil {
			return err
		}
	}
	s, err := m.CreateService(serviceName, exe, mgr.Config{
		DisplayName: serviceDisplayName,
		StartType:   mgr.StartAutomatic,
		Description: serviceDescription,
	}, append([]string{"service", "run"}, serverArgs...)...)
	if err != nil {
		return err
	}
	defer s.Close()
	_ = eventlog.InstallAsEventCreate(serviceName, eventlog.Error|eventlog.Warning|eventlog.Info)
	return nil
}

func uninstallWindowsService() error {
	_ = stopWindowsService()
	m, err := mgr.Connect()
	if err != nil {
		return err
	}
	defer m.Disconnect()
	s, err := m.OpenService(serviceName)
	if err != nil {
		if serviceDoesNotExist(err) {
			return nil
		}
		return err
	}
	defer s.Close()
	if err := s.Delete(); err != nil {
		return err
	}
	_ = eventlog.Remove(serviceName)
	return nil
}

func startWindowsService() error {
	m, err := mgr.Connect()
	if err != nil {
		return err
	}
	defer m.Disconnect()
	s, err := m.OpenService(serviceName)
	if err != nil {
		return err
	}
	defer s.Close()
	if err := s.Start(); err != nil && !errors.Is(err, windows.ERROR_SERVICE_ALREADY_RUNNING) {
		return err
	}
	return nil
}

func stopWindowsService() error {
	m, err := mgr.Connect()
	if err != nil {
		return err
	}
	defer m.Disconnect()
	s, err := m.OpenService(serviceName)
	if err != nil {
		if serviceDoesNotExist(err) {
			return nil
		}
		return err
	}
	defer s.Close()
	status, err := s.Query()
	if err != nil {
		return err
	}
	if status.State == svc.Stopped {
		return nil
	}
	status, err = s.Control(svc.Stop)
	if err != nil {
		if errors.Is(err, windows.ERROR_SERVICE_NOT_ACTIVE) {
			return nil
		}
		return err
	}
	deadline := time.Now().Add(30 * time.Second)
	for status.State != svc.Stopped {
		if time.Now().After(deadline) {
			return fmt.Errorf("timeout waiting for %s to stop", serviceName)
		}
		time.Sleep(500 * time.Millisecond)
		status, err = s.Query()
		if err != nil {
			return err
		}
	}
	return nil
}

func runWindowsService(serverArgs []string) error {
	isService, err := svc.IsWindowsService()
	if err != nil {
		return err
	}
	if !isService {
		return runServer(serverArgs, nil)
	}
	return svc.Run(serviceName, &ntService{serverArgs: serverArgs})
}

type ntService struct {
	serverArgs []string
}

func (s *ntService) Execute(_ []string, requests <-chan svc.ChangeRequest, changes chan<- svc.Status) (bool, uint32) {
	const accepted = svc.AcceptStop | svc.AcceptShutdown
	stop := make(chan struct{})
	done := make(chan error, 1)
	changes <- svc.Status{State: svc.StartPending}
	go func() {
		done <- runServer(s.serverArgs, stop)
	}()
	changes <- svc.Status{State: svc.Running, Accepts: accepted}
	for {
		select {
		case req := <-requests:
			switch req.Cmd {
			case svc.Interrogate:
				changes <- req.CurrentStatus
			case svc.Stop, svc.Shutdown:
				changes <- svc.Status{State: svc.StopPending}
				close(stop)
				err := <-done
				if err != nil {
					return false, 1
				}
				return false, 0
			default:
				changes <- svc.Status{State: svc.Running, Accepts: accepted}
			}
		case err := <-done:
			if err != nil {
				return false, 1
			}
			return false, 0
		}
	}
}

func serviceDoesNotExist(err error) bool {
	return errors.Is(err, windows.ERROR_SERVICE_DOES_NOT_EXIST)
}
